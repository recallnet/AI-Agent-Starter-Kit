import {
  elizaLogger,
  type UUID,
  Service,
  ServiceType,
  stringToUuid,
  IAgentRuntime,
  Content,
} from "@ai16z/eliza";
import duckdb from "duckdb";
import { ParquetReader } from "@dsnp/parquetjs";
import { writeParquetToBuffer } from "./stream.service.js";
import { ChainName, getChain, testnet } from "@recallnet/chains";
import { AccountInfo } from "@recallnet/sdk/account";
import { ListResult } from "@recallnet/sdk/bucket";
import { AccessControlService } from "./acc.service.js";
import {
  RecallClient,
  walletClientFromPrivateKey,
} from "@recallnet/sdk/client";
import { CreditAccount } from "@recallnet/sdk/credit";
import { Address, Hex, parseEther, TransactionReceipt } from "viem";
import { randomUUID } from "crypto";
import { AnyType } from "src/utils.js";

// Interface for Memory objects as used across the application
interface Memory {
  /** Optional unique identifier */
  id?: UUID;
  /** Associated user ID */
  userId: UUID;
  /** Associated agent ID */
  agentId: UUID;
  /** Optional creation timestamp */
  createdAt?: number;
  /** Memory content */
  content: Content;
  /** Optional embedding vector */
  embedding?: number[];
  /** Associated room ID */
  roomId: UUID;
  /** Whether memory is unique */
  unique?: boolean;
  /** Embedding similarity score */
  similarity?: number;
}

export type KnowledgeRecord = {
  id: string;
  userId: string;
  agentId: string;
  content: Content;
  embedding: number[];
  roomId: string;
  createdAt: string;
  knowledgeFileKey: string; // Key in Recall where this record is stored
};

type Result<T = unknown> = {
  result: T;
  meta?: {
    tx?: TransactionReceipt;
  };
};

// Type for the knowledge record as stored in DuckDB
export type DuckDBKnowledgeRecord = {
  id: string;
  userId: string;
  agentId: string;
  content: string;
  embedding: number[];
  roomId: string;
  createdAt: string;
  knowledgeFileKey: string; // Key in Recall where this record is stored
};

// Type for knowledge search results including similarity score
export type KnowledgeSearchResult = Omit<DuckDBKnowledgeRecord, "embedding"> & {
  similarityScore: number;
};

// Type for knowledge items used for formatting and presentation
export type KnowledgeItem = {
  id: string;
  content: Content;
  similarity?: number;
};

// Types for Parquet record format
export type ParquetRecord = {
  userId: string;
  agentId: string;
  userMessage: string;
  log: string;
  embedding: number[];
  timestamp: string;
};

// Type for SQL query results
export type SqlQueryResult = {
  [key: string]: string | number | boolean | number[] | null;
};

// Specify detailed object bucket query response
export type BucketQueryResponse = {
  result?: {
    objects: Array<{
      key: string;
      size: number;
      lastModified: string;
    }>;
  };
};

// Load environment variables with detailed logging
const privateKey = process.env.RECALL_PRIVATE_KEY as Hex;
const envAlias = process.env.RECALL_BUCKET_ALIAS as string;
const envPrefix = process.env.RECALL_MEMORY_PREFIX as string;
const network = process.env.RECALL_NETWORK as string;

// Add debug logging for environment variables
elizaLogger.info("[RecallService] Environment configuration:", {
  RECALL_PRIVATE_KEY: privateKey ? "[REDACTED]" : undefined,
  RECALL_BUCKET_ALIAS: envAlias,
  RECALL_MEMORY_PREFIX: envPrefix,
  RECALL_NETWORK: network,
});

type SqlParam = string | number | boolean | number[] | null;

export class RecallService extends Service {
  static get serviceType(): ServiceType {
    elizaLogger.info("[RecallService] Getting RecallService.serviceType");
    return "recall" as ServiceType;
  }

  private client: RecallClient;
  private runtime: IAgentRuntime;
  private alias: string;
  private prefix: string;
  private db: duckdb.Connection;
  private accessControlService: AccessControlService | undefined;
  private processedFiles: Set<string> = new Set();
  private isInitialized: boolean = false;

  // Define timeout constants for easier configuration
  private static readonly DEFAULT_TIMEOUT = 30000; // 30 seconds
  private static readonly BUCKET_OPERATION_TIMEOUT = 15000; // 15 seconds
  private static readonly MAX_RETRY_ATTEMPTS = 3;
  private static readonly RETRY_DELAY_MS = 1000;

  getInstance(): RecallService {
    elizaLogger.info("[RecallService] RecallService.getInstance() called");
    return this;
  }

  constructor(_runtime: IAgentRuntime) {
    super();
    elizaLogger.info("[RecallService] RecallService constructor called");
    try {
      elizaLogger.info(
        "[RecallService] RecallService super() constructor completed"
      );
      this.runtime = _runtime;
      elizaLogger.info(
        "[RecallService] RecallService constructor runtime assigned",
        {
          runtimeExists: !!_runtime,
          runtimeType: _runtime ? typeof _runtime : "undefined",
        }
      );
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error in RecallService constructor: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
      throw error;
    }
  }

  async initialize(runtime: IAgentRuntime): Promise<void> {
    elizaLogger.info("[RecallService] RecallService.initialize() called", {
      hasRuntime: !!runtime,
      hasThisRuntime: !!this.runtime,
    });

    try {
      // Guard against multiple initializations
      if (this.isInitialized) {
        elizaLogger.warn(
          "[RecallService] RecallService already initialized, skipping"
        );
        return;
      }

      // Validate environment variables
      this.validateEnvironmentVariables();

      // Use runtime from parameter if provided, fallback to constructor runtime
      if (runtime) {
        elizaLogger.info(
          "[RecallService] Using runtime from initialize() parameter"
        );
        this.runtime = runtime;
      } else if (!this.runtime) {
        elizaLogger.error(
          "[RecallService] No runtime available for initialization"
        );
        throw new Error(
          "[RecallService] No runtime available for initialization"
        );
      }

      elizaLogger.info("[RecallService] RecallService initialization started");

      // Set up blockchain connection
      await this.setupBlockchainConnection();

      // Initialize DuckDB
      await this.initializeDatabase();

      // Load processed files into memory
      elizaLogger.info("[RecallService] Loading processed files into memory");
      try {
        await this.loadProcessedFiles();
        elizaLogger.info(
          `[RecallService] Loaded ${this.processedFiles.size} processed files`
        );
      } catch (loadError) {
        elizaLogger.error(
          `[RecallService] Error loading processed files: ${loadError.message}`
        );
        throw loadError;
      }

      // Initialize accessControlService
      await this.initializeAccessControlService();
      elizaLogger.info("[RecallService] AccessControlService initialized");

      this.isInitialized = true;
      elizaLogger.success(
        "[RecallService] RecallService initialized successfully"
      );
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error initializing RecallService: ${error.message}`,
        {
          error,
          stack: error.stack,
          runtimeExists: !!this.runtime,
        }
      );
      throw error;
    }
  }

  /**
   * Validates required environment variables
   * @throws Error if required environment variables are missing
   */
  private validateEnvironmentVariables(): void {
    // Validate environment variables
    if (!privateKey) {
      elizaLogger.error("[RecallService]: RECALL_PRIVATE_KEY is required");
      throw new Error("[RecallService]: RECALL_PRIVATE_KEY is required");
    }
    if (!envAlias) {
      elizaLogger.error("[RecallService]: RECALL_BUCKET_ALIAS is required");
      throw new Error("[RecallService]: RECALL_BUCKET_ALIAS is required");
    }
    if (!envPrefix) {
      elizaLogger.error("[RecallService]: RECALL_MEMORY_PREFIX is required");
      throw new Error("[RecallService]: RECALL_MEMORY_PREFIX is required");
    }
  }

  /**
   * Sets up blockchain connection and client
   */
  private async setupBlockchainConnection(): Promise<void> {
    // Set up blockchain connection
    elizaLogger.info(
      `[RecallService] Setting up blockchain connection with network: ${network || "testnet"}`
    );
    const chain = network ? getChain(network as ChainName) : testnet;
    elizaLogger.info("[RecallService] Creating wallet client from private key");
    const wallet = walletClientFromPrivateKey(privateKey, chain);
    elizaLogger.info("[RecallService] Creating RecallClient");
    this.client = new RecallClient({ walletClient: wallet });

    // Set configuration values
    this.alias = envAlias;
    this.prefix = envPrefix;
    elizaLogger.info(
      `[RecallService] RecallService configured with alias: ${this.alias}, prefix: ${this.prefix}`
    );
  }

  /**
   * Initializes DuckDB database and creates schema
   */
  private async initializeDatabase(): Promise<void> {
    // Initialize DuckDB
    elizaLogger.info("[RecallService] Initializing DuckDB in-memory database");
    try {
      const db = new duckdb.Database(":memory:"); // In-memory DB for performance
      this.db = db.connect();
      elizaLogger.info("[RecallService] DuckDB connection established");
    } catch (dbError) {
      elizaLogger.error(
        `[RecallService] Failed to initialize DuckDB: ${dbError.message}`,
        {
          error: dbError,
          stack: dbError.stack,
        }
      );
      throw dbError;
    }

    // Create database schema
    elizaLogger.info("[RecallService] Creating DuckDB schema");
    try {
      await this.createDatabaseSchema();
      elizaLogger.info("[RecallService] DuckDB schema creation completed");
    } catch (schemaError) {
      elizaLogger.error(
        `[RecallService] Failed to create schema: ${schemaError.message}`
      );
      throw schemaError;
    }
  }

  /**
   * Creates the database schema in DuckDB
   */
  private async createDatabaseSchema(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.db.exec(
        `
        -- ✅ Load Vector Similarity Search (VSS) Extension
        INSTALL vss;
        LOAD vss;
    
        -- ✅ Create Knowledge Table with Fixed Embedding Size
        CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY,
            userId TEXT,
            agentId TEXT,
            content TEXT,
            embedding FLOAT[1536], -- ⚠️ Ensure your embeddings match this size
            roomId TEXT,
            createdAt TEXT,
            knowledgeFileKey TEXT,
            UNIQUE(id, knowledgeFileKey)
        );
    
        -- ✅ Create HNSW Index for Fast Vector Similarity Search
        CREATE INDEX IF NOT EXISTS knowledge_hnsw_index 
        ON knowledge USING HNSW (embedding)
        WITH (metric = 'cosine');
    
        -- ✅ Create Processed Files Table
        CREATE TABLE IF NOT EXISTS processed_files (
            fileKey TEXT PRIMARY KEY,
            processedAt TEXT
        );
        `,
        (err) => {
          if (err) {
            elizaLogger.error(
              `[RecallService]: ⛔ Error creating schema: ${err.message}`,
              {
                error: err,
                stack: err.stack,
              }
            );
            reject(err);
          } else {
            elizaLogger.info("[RecallService]: ✅ Schema created successfully");
            resolve();
          }
        }
      );
    });
  }

  /**
   * Initialize the AccessControlService during RecallService initialization
   * Add this to your initialize() method
   */
  async initializeAccessControlService(): Promise<boolean> {
    try {
      elizaLogger.info(
        "[RecallService] Initializing AccessControlService for encryption/decryption"
      );
      this.accessControlService = AccessControlService.getInstance();
      await this.accessControlService.start();

      if (!this.accessControlService.isConfigured()) {
        elizaLogger.warn(
          "[RecallService] AccessControlService is not properly configured. Content will not be encrypted."
        );
        return false;
      } else {
        elizaLogger.info(
          "[RecallService] AccessControlService initialized successfully. Content encryption is enabled."
        );
        return true;
      }
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error initializing AccessControlService: ${error.message}`
      );
      return false;
    }
  }

  /**
   * Utility function to handle timeouts for async operations.
   * @param promise The promise to execute.
   * @param timeoutMs The timeout in milliseconds.
   * @param operationName The name of the operation for logging.
   * @returns The result of the promise.
   */
  async withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number,
    operationName: string
  ): Promise<T> {
    let timeoutId: NodeJS.Timeout;

    const timeoutPromise = new Promise<T>((_, reject) => {
      timeoutId = setTimeout(() => {
        reject(
          new Error(`${operationName} operation timed out after ${timeoutMs}ms`)
        );
      }, timeoutMs);
    });

    try {
      const result = await Promise.race([promise, timeoutPromise]);
      clearTimeout(timeoutId!);
      return result;
    } catch (error) {
      clearTimeout(timeoutId!);
      throw error;
    }
  }

  /**
   * Executes a SQL query and returns the results
   * @param query SQL query to execute
   * @param params Parameters for the query
   * @returns Promise resolving to query results
   */
  private async executeSql<T extends Record<string, unknown>>(
    query: string,
    ...params: SqlParam[]
  ): Promise<T[]> {
    return new Promise((resolve, reject) => {
      this.db.all(
        query,
        ...params,
        (err: Error | null, rows: duckdb.TableData) => {
          if (err) {
            elizaLogger.error(
              `[RecallService]: Error executing SQL: ${err.message}`,
              {
                error: err,
                query,
                params: JSON.stringify(params),
              }
            );
            reject(err);
            return;
          }
          resolve(rows as unknown as T[]);
        }
      );
    });
  }

  /**
   * Executes a SQL statement that doesn't return results
   * @param statement SQL statement to execute
   * @param params Parameters for the statement
   * @returns Promise resolving when the statement completes
   */
  private async executeSqlStatement(
    statement: string,
    ...params: SqlParam[]
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.run(statement, ...params, (err) => {
        if (err) {
          elizaLogger.error(
            `[RecallService]: Error executing SQL statement: ${err.message}`,
            {
              error: err,
              statement,
              params: JSON.stringify(params),
            }
          );
          reject(err);
          return;
        }
        resolve();
      });
    });
  }

  /**
   * Loads processed files from DuckDB into memory.
   * @returns A promise that resolves when the files are loaded.
   */
  private async loadProcessedFiles(): Promise<void> {
    elizaLogger.info("[RecallService]: Loading processed files from DuckDB");
    try {
      const rows = await this.executeSql<{ fileKey: string }>(
        "SELECT fileKey FROM processed_files;"
      );
      this.processedFiles = new Set(rows.map((row) => row.fileKey));
      elizaLogger.info(
        `[RecallService]: Loaded ${this.processedFiles.size} processed files from database`
      );
    } catch (error) {
      elizaLogger.error(
        `[RecallService]: Error loading processed files: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
      throw error;
    }
  }

  /**
   * Marks a file as processed in both DuckDB and memory
   * @param fileKey The key of the file to mark as processed.
   * @returns A promise that resolves when the file is marked as processed.
   */
  private async markFileAsProcessed(fileKey: string): Promise<void> {
    if (this.processedFiles.has(fileKey)) {
      return; // Already processed
    }

    try {
      await this.executeSqlStatement(
        "INSERT INTO processed_files (fileKey, processedAt) VALUES (?, ?);",
        fileKey,
        new Date().toISOString()
      );
      this.processedFiles.add(fileKey);
      elizaLogger.debug(`[RecallService] Marked file as processed: ${fileKey}`);
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error marking file as processed: ${error.message}`,
        {
          error,
          stack: error.stack,
          fileKey,
        }
      );
      throw error;
    }
  }

  /**
   * Gets the account information for the current user.
   * @returns The account information.
   */
  public async getAccountInfo(): Promise<AccountInfo | undefined> {
    try {
      const info = await this.client.accountManager().info();
      return info.result;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error getting account info: ${error.message}`
      );
      throw error;
    }
  }

  /**
   * Lists all buckets in Recall.
   * @returns The list of buckets.
   */
  public async listBuckets(): Promise<ListResult | undefined> {
    try {
      const info = await this.client.bucketManager().list();
      return info.result;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error listing buckets: ${error.message}`
      );
      throw error;
    }
  }

  /**
   * Gets the credit information for the account.
   * @returns The credit information.
   */
  public async getCreditInfo(): Promise<CreditAccount | undefined> {
    try {
      const info = await this.client.creditManager().getAccount();
      return info.result;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error getting credit info: ${error.message}`
      );
      throw error;
    }
  }

  /**
   * Buys credit for the account.
   * @param amount The amount of credit to buy.
   * @returns The result of the buy operation.
   */
  public async buyCredit(amount: string): Promise<Result> {
    try {
      const info = await this.client.creditManager().buy(parseEther(amount));
      return info;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error buying credit: ${error.message}`
      );
      throw error;
    }
  }

  /**
   * Gets or creates a knowledge bucket in Recall.
   * @param bucketAlias The alias of the bucket to retrieve or create.
   * @returns The address of the knowledge bucket.
   */
  public async getOrCreateBucket(bucketAlias: string): Promise<Address> {
    try {
      elizaLogger.info(
        `[RecallService] Looking for bucket with alias: ${bucketAlias}`
      );

      // Try to find the bucket by alias
      const buckets = await this.client.bucketManager().list();
      if (buckets?.result) {
        const bucket = buckets.result.find(
          (b) => b.metadata?.alias === bucketAlias
        );
        if (bucket) {
          elizaLogger.info(
            `[RecallService] Found existing bucket "${bucketAlias}" at ${bucket.addr}`
          );
          return bucket.addr; // Return existing bucket address
        } else {
          elizaLogger.info(
            `[RecallService] Bucket with alias "${bucketAlias}" not found, creating a new one.`
          );
        }
      }

      // Ensure bucketAlias is correctly passed during creation
      const query = await this.client.bucketManager().create({
        metadata: { alias: bucketAlias },
      });

      const newBucket = query.result;
      if (!newBucket) {
        elizaLogger.error(
          `[RecallService] Failed to create new bucket with alias: ${bucketAlias}`
        );
        throw new Error(
          `[RecallService] Failed to create bucket: ${bucketAlias}`
        );
      }

      elizaLogger.info(
        `[RecallService] Successfully created new bucket "${bucketAlias}" at ${newBucket.bucket}`
      );
      return newBucket.bucket;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error in getOrCreateBucket: ${error.message}`
      );
      throw error;
    }
  }

  /**
   * Adds an object to a bucket.
   * @param bucket The address of the bucket.
   * @param key The key under which to store the object.
   * @param data The data to store (string, File, or Uint8Array).
   * @param options Optional parameters.
   * @returns A Result object.
   */
  public async addObject(
    bucket: Address,
    key: string,
    data: string | File | Uint8Array,
    options?: { overwrite?: boolean }
  ): Promise<Result> {
    try {
      const info = await this.client.bucketManager().add(bucket, key, data, {
        overwrite: options?.overwrite ?? false,
      });
      return info;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error adding object: ${error.message}`
      );
      throw error;
    }
  }

  /**
   * Gets an object from a bucket.
   * @param bucket The address of the bucket.
   * @param key The key under which the object is stored.
   * @returns The data stored under the specified key.
   */
  public async getObject(
    bucket: Address,
    key: string
  ): Promise<Uint8Array | undefined> {
    try {
      const info = await this.client.bucketManager().get(bucket, key);
      return info.result;
    } catch (error) {
      elizaLogger.warn(
        `[RecallService] Error getting object: ${error.message}`
      );
      throw error;
    }
  }

  /**
   * Validates a memory's embedding and logs details about its state.
   * @param memory The memory to validate
   * @param context A string describing where the validation is happening
   * @returns true if the embedding is valid, false otherwise
   */
  private validateEmbedding(
    memory: Memory | null | undefined,
    context: string
  ): boolean {
    if (!memory) {
      elizaLogger.debug(
        `[RecallService] Validation failed at ${context}: memory is null or undefined`
      );
      return false;
    }

    const validation = {
      memoryExists: true,
      hasEmbeddingProperty: "embedding" in memory,
      embeddingDefined: !!memory.embedding,
      isArray: false,
      hasLength: false,
      isNumberArray: false,
      context,
      memoryId: memory.id || "unknown",
    };

    try {
      // Handle string embeddings by parsing them
      if (typeof memory.embedding === "string") {
        try {
          memory.embedding = JSON.parse(memory.embedding as unknown as string);
        } catch (e) {
          elizaLogger.error(
            `[RecallService] Failed to parse embedding string at ${context}: ${e.message}`
          );
          return false;
        }
      }

      // Update validation status
      validation.isArray = Array.isArray(memory.embedding) || false;
      validation.hasLength =
        (validation.isArray &&
          memory.embedding &&
          memory.embedding.length > 0) ||
        false;
      validation.isNumberArray =
        (validation.hasLength &&
          memory.embedding &&
          memory.embedding.every((n) => typeof n === "number")) ||
        false;

      const isValid =
        validation.memoryExists &&
        validation.hasEmbeddingProperty &&
        validation.embeddingDefined &&
        validation.isArray &&
        validation.hasLength &&
        validation.isNumberArray;

      // Log validation results
      if (!isValid) {
        elizaLogger.debug(
          `[RecallService] Embedding validation failed at ${context} for memory ${validation.memoryId}`,
          {
            validationResults: validation,
            failureReason: {
              noMemory: !validation.memoryExists,
              noEmbeddingProperty: !validation.hasEmbeddingProperty,
              embeddingUndefined: !validation.embeddingDefined,
              notArray: !validation.isArray,
              emptyArray: !validation.hasLength,
              notNumberArray: !validation.isNumberArray,
            },
          }
        );
      }

      return isValid;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error during embedding validation at ${context}: ${error.message}`
      );
      return false;
    }
  }

  /**
   * Prepares memories for storage by ensuring they have embeddings.
   * @param memories Array of memories to prepare
   * @returns Array of prepared memories with embeddings
   */
  private async prepareMemoriesForStorage(
    memories: Memory[]
  ): Promise<Memory[]> {
    const preparedMemories: Memory[] = [];

    for (const memory of memories) {
      try {
        // Initial validation check
        const initialValidation = this.validateEmbedding(
          memory,
          "initial check"
        );
        elizaLogger.info(`Processing memory ${memory.id}`, {
          hasValidEmbedding: initialValidation,
          embeddingLength: memory.embedding?.length,
        });

        if (initialValidation && memory.embedding) {
          // Check that embedding exists
          // Memory already has valid embedding, make a deep copy
          const preparedMemory = {
            ...memory,
            embedding: Array.isArray(memory.embedding)
              ? [...memory.embedding]
              : [],
          };

          // Verify copy was successful
          if (this.validateEmbedding(preparedMemory, "after copy")) {
            preparedMemories.push(preparedMemory);
            elizaLogger.info(
              `[RecallService] Using existing embedding for memory ${memory.id}`,
              {
                embeddingLength: preparedMemory.embedding.length,
              }
            );
            continue;
          } else {
            elizaLogger.warn(
              `[RecallService] Failed to copy embedding for memory ${memory.id}, will try to generate new one`
            );
          }
        }

        // Need to generate new embedding
        elizaLogger.info(
          `[RecallService] Generating new embedding for memory ${memory.id}`
        );

        // Create a clean copy without any existing embedding
        const memoryForEmbedding = {
          ...memory,
          embedding: undefined,
          id: memory.id || stringToUuid(randomUUID()),
        };

        // Try to generate embedding with retries
        let attemptsLeft = RecallService.MAX_RETRY_ATTEMPTS;
        let embeddingSuccess = false;
        let generatedMemory: Memory = memoryForEmbedding;

        while (attemptsLeft > 0 && !embeddingSuccess) {
          try {
            const result =
              await this.runtime.messageManager.addEmbeddingToMemory(
                memoryForEmbedding
              );

            if (result && this.validateEmbedding(result, "after generation")) {
              generatedMemory = result;
              embeddingSuccess = true;
              break;
            }

            attemptsLeft--;
            if (attemptsLeft > 0) {
              elizaLogger.warn(
                `[RecallService] Invalid embedding generated for memory ${memory.id}, ${attemptsLeft} attempts remaining`
              );
              await new Promise((resolve) =>
                setTimeout(resolve, RecallService.RETRY_DELAY_MS)
              );
            }
          } catch (embedError) {
            attemptsLeft--;
            if (attemptsLeft > 0) {
              elizaLogger.warn(
                `[RecallService] Error generating embedding for memory ${memory.id}, ${attemptsLeft} attempts remaining: ${embedError.message}`
              );
              await new Promise((resolve) =>
                setTimeout(resolve, RecallService.RETRY_DELAY_MS)
              );
            } else {
              throw embedError;
            }
          }
        }

        // Check if we succeeded in generating a valid embedding
        if (
          embeddingSuccess &&
          this.validateEmbedding(generatedMemory, "final validation")
        ) {
          preparedMemories.push(generatedMemory);
          elizaLogger.info(
            `[RecallService] Successfully added new embedding to memory ${memory.id}`,
            {
              embeddingLength: generatedMemory.embedding?.length,
            }
          );
        } else {
          throw new Error(
            `[RecallService] Failed to generate valid embedding for memory ${memory.id} after all attempts`
          );
        }
      } catch (error) {
        elizaLogger.error(
          `[RecallService] Failed to prepare memory ${memory.id}: ${error.message}`,
          {
            error,
            stack: error.stack,
            memoryContent:
              typeof memory.content === "string"
                ? "string content"
                : JSON.stringify(memory.content),
          }
        );
      }
    }

    // Log overall results
    elizaLogger.info(`[RecallService] Memory preparation complete`, {
      total: memories.length,
      prepared: preparedMemories.length,
      successRate: `${((preparedMemories.length / memories.length) * 100).toFixed(1)}%`,
    });

    // Final validation of all prepared memories
    const invalidMemories = preparedMemories.filter(
      (memory) => !this.validateEmbedding(memory, "final batch check")
    );

    if (invalidMemories.length > 0) {
      const error = new Error(
        `[RecallService] ${invalidMemories.length} memories have invalid embeddings after preparation`
      );
      elizaLogger.error(error.message, {
        invalidMemoryIds: invalidMemories.map((m) => m.id),
      });
      throw error;
    }

    return preparedMemories;
  }

  /**
   * Transforms Memory objects to ParquetRecord format for storage
   * @param memories Array of memory objects
   * @returns Array of ParquetRecord objects
   */
  private async transformMemoriesToParquetRecords(
    memories: Memory[]
  ): Promise<ParquetRecord[]> {
    return memories.map((memory) => {
      // Log the memory structure before transformation
      elizaLogger.debug(`[RecallService] Memory pre-transform:`, {
        id: memory.id,
        hasEmbedding: !!memory.embedding,
        embeddingLength: memory.embedding?.length,
      });

      // Extract text from content object or handle string content
      const text =
        typeof memory.content === "string"
          ? memory.content
          : memory.content.text || "";

      elizaLogger.debug(
        `[RecallService] Creating record from memory ID=${memory.id}, roomId=${memory.roomId}`
      );

      if (!memory.embedding || !Array.isArray(memory.embedding)) {
        throw new Error(
          `[RecallService] Memory ${memory.id} has invalid embedding during transformation`
        );
      }

      const record: ParquetRecord = {
        userId: memory.userId || "",
        agentId: memory.agentId || "",
        userMessage: text, // Store content as userMessage
        log: text, // Also store in log for redundancy
        embedding: [...memory.embedding], // Make a copy of the embedding array
        timestamp: memory.createdAt
          ? new Date(memory.createdAt).toISOString()
          : new Date().toISOString(),
      };

      // Log the record after transformation
      elizaLogger.debug(`[RecallService] Knowledge record created:`, {
        userId: record.userId,
        hasEmbedding: Array.isArray(record.embedding),
        embeddingLength: record.embedding?.length,
        recordKeys: Object.keys(record),
      });

      return record;
    });
  }

  /**
   * Creates a Parquet buffer from memory objects
   * @param memories Array of memory objects to convert to Parquet
   * @returns Buffer containing Parquet data
   */
  private async createParquetBuffer(memories: Memory[]): Promise<Buffer> {
    // First transform memories to the proper record format
    const records = await this.transformMemoriesToParquetRecords(memories);

    // Verify all records have valid embeddings before proceeding
    const withoutEmbeddings = records.filter(
      (record) =>
        !record.embedding ||
        !Array.isArray(record.embedding) ||
        record.embedding.length === 0
    );

    if (withoutEmbeddings.length > 0) {
      elizaLogger.error(
        `[RecallService] ${withoutEmbeddings.length} records missing embeddings before Parquet creation`,
        {
          records: withoutEmbeddings.map((r) => ({
            userId: r.userId,
            hasEmbedding: !!r.embedding,
            embeddingLength: r.embedding?.length,
          })),
        }
      );
      throw new Error("[RecallService] Some records are missing embeddings");
    }

    // More detailed logging around Parquet creation
    elizaLogger.info(
      `[RecallService] Attempting to create Parquet buffer for ${records.length} records with structure:`,
      {
        sampleKeys: Object.keys(records[0]),
        hasSampleEmbedding: !!records[0].embedding,
        sampleEmbeddingLength: records[0].embedding?.length,
      }
    );

    // Create the Parquet buffer
    const parquetBuffer = await writeParquetToBuffer(records);

    if (!parquetBuffer) {
      throw new Error("[RecallService] Failed to generate Parquet buffer");
    }

    if (parquetBuffer.length === 0) {
      throw new Error("[RecallService] Generated Parquet buffer is empty");
    }

    elizaLogger.info(
      `[RecallService] Successfully generated Parquet buffer of ${parquetBuffer.length} bytes`
    );

    return parquetBuffer;
  }

  /**
   * Stores a batch of knowledge to Recall.
   * @param bucketAddress The address of the bucket to store knowledge.
   * @param batch The batch of memories to store.
   * @returns The key under which the knowledge was stored.
   */
  async storeBatchToRecall(
    bucketAddress: Address,
    batch: Memory[]
  ): Promise<string | undefined> {
    try {
      const timestamp = Date.now();
      const nextKnowledgeKey = `${this.prefix}${timestamp}.parquet`;

      // Ensure all memories have embeddings
      elizaLogger.info(
        `[RecallService] Preparing ${batch.length} memories for storage`
      );
      const preparedMemories = await this.prepareMemoriesForStorage(batch);

      // Verify all prepared memories have embeddings
      const missingEmbeddings = preparedMemories.filter(
        (memory) =>
          !memory.embedding ||
          !Array.isArray(memory.embedding) ||
          memory.embedding.length === 0
      );

      if (missingEmbeddings.length > 0) {
        elizaLogger.error(
          `[RecallService] ${missingEmbeddings.length} memories still missing embeddings after preparation`,
          {
            memoryIds: missingEmbeddings.map((m) => m.id),
          }
        );
        return undefined;
      }

      if (preparedMemories.length === 0) {
        elizaLogger.warn(
          "[RecallService] No valid memories to store after preparation."
        );
        return undefined;
      }

      elizaLogger.info(
        `[RecallService] Transforming ${preparedMemories.length} memories to knowledge format`
      );

      // Encrypt memory content before storing
      const encryptedMemories = await Promise.all(
        preparedMemories.map(async (memory) => {
          let finalMemory;
          if (!this.accessControlService) {
            elizaLogger.warn(
              "[RecallService] AccessControlService is unavailable, storing data without encryption."
            );
            finalMemory = memory;
          } else {
            finalMemory =
              await this.accessControlService.prepareMemoryForStorage(memory);
            if (!finalMemory) {
              elizaLogger.error(
                `[RecallService] Failed to encrypt memory ${memory.id}`
              );
              return memory;
            }
          }
          return finalMemory;
        })
      );

      try {
        const parquetBuffer = await this.createParquetBuffer(encryptedMemories);

        if (!parquetBuffer) {
          elizaLogger.error(
            "[RecallService] writeParquetToBuffer returned undefined"
          );
          return undefined;
        }

        if (parquetBuffer.length === 0) {
          elizaLogger.error(
            "[RecallService] Generated Parquet file is empty. Skipping upload."
          );
          return undefined;
        }

        elizaLogger.info(
          `[RecallService] Uploading Parquet data to bucket ${bucketAddress} with key ${nextKnowledgeKey}`
        );
        const addObject = await this.withTimeout(
          this.client.bucketManager().add(
            bucketAddress,
            nextKnowledgeKey,
            Uint8Array.from(parquetBuffer) // Convert Buffer to Uint8Array
          ),
          RecallService.DEFAULT_TIMEOUT,
          "Recall batch storage"
        );

        if (!addObject?.meta?.tx) {
          elizaLogger.error(
            "[RecallService] ❌ Recall API returned invalid response for batch storage",
            {
              response: JSON.stringify(addObject),
              bucket: bucketAddress,
              key: nextKnowledgeKey,
              batchSize: preparedMemories.length,
            }
          );
          return undefined;
        }

        elizaLogger.info(
          `[RecallService] Successfully stored batch of ${encryptedMemories.length} records at key: ${nextKnowledgeKey}`
        );
        return nextKnowledgeKey;
      } catch (parquetError) {
        elizaLogger.error(
          `[RecallService] Parquet generation/upload error: ${parquetError.message}`,
          {
            error: parquetError,
            stack: parquetError.stack,
          }
        );
        return undefined;
      }
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error storing knowledge as Parquet in Recall: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
      return undefined;
    }
  }

  /**
   * Write a single memory to Recall as knowledge
   * @param message Memory object to store
   * @returns Key of the stored knowledge or undefined if operation failed
   */
  async writeKnowledgeToRecall(message: Memory): Promise<string | undefined> {
    try {
      const bucketAddress = await this.withTimeout(
        this.getOrCreateBucket(this.alias),
        RecallService.BUCKET_OPERATION_TIMEOUT,
        "Get/Create bucket"
      );

      elizaLogger.info(
        `[RecallService] 📂 Processing message ID=${message.id}`
      );

      const knowledgeFileKey = await this.storeBatchToRecall(
        bucketAddress,
        [message] // Syncing only the input message
      );

      if (knowledgeFileKey) {
        await this.insertKnowledgeIntoDuckDB(message, knowledgeFileKey);
        elizaLogger.success(
          `[RecallService] ✅ Successfully synced message ID=${message.id}`
        );
        return knowledgeFileKey;
      } else {
        elizaLogger.warn(
          `[RecallService] ⚠️ Failed to sync message ID=${message.id} - will retry on next sync`
        );
        return undefined;
      }
    } catch (error) {
      if (error.message.includes("timed out")) {
        elizaLogger.error(
          `[RecallService] ⏳ Recall sync operation timed out: ${error.message}`
        );
        return undefined;
      } else {
        elizaLogger.error(
          `[RecallService] ❌ Error in writeKnowledgeToRecall: ${error.message}`
        );
        return undefined;
      }
    }
  }

  /**
   * Inserts a knowledge record into DuckDB.
   * @param memory The memory to insert.
   * @param knowledgeFileKey The key of the knowledge file in Recall.
   * @returns A promise that resolves when the knowledge is inserted.
   */
  private async insertKnowledgeIntoDuckDB(
    memory: Memory,
    knowledgeFileKey: string
  ): Promise<void> {
    if (
      !memory.embedding ||
      !Array.isArray(memory.embedding) ||
      memory.embedding.length === 0
    ) {
      elizaLogger.warn(
        `[RecallService] Memory ${memory.id} has no embedding, skipping insertion into DuckDB`
      );
      return;
    }

    // Debug the values we're about to insert
    elizaLogger.debug(
      `[RecallService] Preparing to insert memory into DuckDB: ID=${memory.id}, file=${knowledgeFileKey}`
    );

    const embeddingArray = memory.embedding.map((num) =>
      parseFloat(String(num))
    );
    const contentStr =
      typeof memory.content === "string"
        ? memory.content
        : JSON.stringify(memory.content);
    const createdAtStr = memory.createdAt
      ? new Date(memory.createdAt).toISOString()
      : new Date().toISOString();
    const memoryId = memory.id || stringToUuid(randomUUID());

    try {
      // Use the reusable SQL execution method
      await this.executeSqlStatement(
        `INSERT INTO knowledge 
         VALUES (?, ?, ?, ?, CAST(? AS FLOAT[]), ?, ?, ?)
         ON CONFLICT (id, knowledgeFileKey) DO NOTHING;`,
        memoryId,
        memory.userId,
        memory.agentId,
        contentStr,
        JSON.stringify(embeddingArray),
        memory.roomId,
        createdAtStr,
        knowledgeFileKey
      );

      elizaLogger.debug(
        `[RecallService] Successfully inserted memory ${memoryId} into DuckDB`
      );
    } catch (error) {
      if (!error.message.includes("UNIQUE constraint")) {
        elizaLogger.error(
          `[RecallService] Error inserting knowledge: ${error.message}`,
          {
            error,
            stack: error.stack,
          }
        );
        throw error;
      }
    }
  }

  /**
   * Query knowledge across multiple rooms with relevance to a given text.
   * @param queryText The text to search for relevant knowledge.
   * @param limit The maximum number of results to return.
   * @param threshold The minimum similarity threshold.
   * @returns An array of knowledge items formatted for use in the runtime.
   */
  // Original code for queryKnowledge
  async queryKnowledge(
    queryText: Memory,
    limit = 10,
    threshold = 0.7
  ): Promise<KnowledgeItem[]> {
    try {
      elizaLogger.info("[RecallService] 📡 queryKnowledge() called", {
        queryTextContent: queryText.content,
        limit,
        threshold,
      });

      // Generate embedding for the query text
      const queryEmbedding =
        queryText.embedding ||
        (await (
          await this.runtime.messageManager.addEmbeddingToMemory(queryText)
        ).embedding);

      if (!queryEmbedding || queryEmbedding.length === 0) {
        elizaLogger.error(
          "[RecallService] ❌ Failed to generate embedding for query text."
        );
        return [];
      }

      elizaLogger.info("[RecallService] 🔍 Query Embedding Generated", {
        queryEmbeddingExists: !!queryEmbedding,
        embeddingLength: queryEmbedding.length,
        first10Values: queryEmbedding.slice(0, 10),
      });

      // Convert embedding to DuckDB array format
      const queryEmbeddingArray = `ARRAY[${[...queryEmbedding].join(",")}]::FLOAT[${queryEmbedding.length}]`;

      // Construct optimized SQL query using array_cosine_distance
      const query = `
      SELECT id, userId, agentId, content, roomId, createdAt, knowledgeFileKey,
        1 - array_cosine_distance(embedding, ${queryEmbeddingArray}) AS similarity
      FROM knowledge
      WHERE 1 - array_cosine_distance(embedding, ${queryEmbeddingArray}) > ${threshold}
      ORDER BY similarity DESC
      LIMIT ${limit};
    `;

      elizaLogger.info(
        "[RecallService] 📝 [RecallService] Executing SQL Query"
      );

      // Execute SQL query in DuckDB
      const searchResults = await this.executeSql<SqlQueryResult>(query);

      elizaLogger.info(
        `[RecallService]: ${searchResults.length} similar results found`
      );

      // Check if any results were found
      if (!searchResults || searchResults.length === 0) {
        elizaLogger.warn("[RecallService] ⚠️ No similar knowledge found.");
        return [];
      }

      // Transform results into KnowledgeItem format
      return this.transformSearchResultsToKnowledgeItems(searchResults);
    } catch (error) {
      elizaLogger.error("[RecallService] ❌ Error in queryKnowledge:", {
        error: error.message,
        stack: error.stack,
      });
      return [];
    }
  }

  /**
   * Transform search results from DuckDB to KnowledgeItem format
   * @param searchResults Array of search results from DuckDB
   * @returns Array of KnowledgeItem objects
   */
  private transformSearchResultsToKnowledgeItems(
    searchResults: SqlQueryResult[]
  ): KnowledgeItem[] {
    return searchResults.map((result) => {
      let contentObj: Content;

      try {
        if (typeof result.content === "string") {
          try {
            // Try to parse content as JSON
            contentObj = JSON.parse(result.content as string);

            // If parsed but doesn't have text property, convert to proper format
            if (!contentObj.text && typeof contentObj === "string") {
              contentObj = { text: contentObj as unknown as string };
            }
          } catch (e) {
            // If can't parse as JSON, treat as plain text
            contentObj = {
              text: String(result.content ?? "No content available"),
            };
          }
        } else {
          // Handle non-string content
          contentObj = {
            text: String(result.content ?? "No content available"),
          };
        }
      } catch (e) {
        // Fallback for any parsing issues
        contentObj = { text: String(result.content ?? "No content available") };
      }

      return {
        id: result.id as string,
        content: contentObj,
        similarity: parseFloat(result.similarity as string),
      };
    });
  }

  /**
   * Formats knowledge items into a string for inclusion in the agent context.
   * @param items The knowledge items to format.
   * @returns A formatted string of knowledge items.
   */
  formatKnowledgeForContext(items: KnowledgeItem[]): string {
    if (!items || items.length === 0) {
      return "";
    }

    // Sort by similarity if available
    const sortedItems = [...items].sort((a, b) => {
      if (a.similarity !== undefined && b.similarity !== undefined) {
        return b.similarity - a.similarity;
      }
      return 0;
    });

    // Format each item with its content
    return sortedItems
      .map((item, index) => {
        const content = item.content;
        const text = content.text || "";

        // Format with additional context if available
        let formattedItem = `KNOWLEDGE ITEM ${index + 1}:\n${text}\n`;

        if (content.source) {
          formattedItem += `Source: ${content.source}\n`;
        }

        if (content.url) {
          formattedItem += `URL: ${content.url}\n`;
        }

        if (item.similarity !== undefined) {
          formattedItem += `Relevance: ${(item.similarity * 100).toFixed(1)}%\n`;
        }

        return formattedItem;
      })
      .join("\n");
  }

  /**
   * Retrieves and processes knowledge files from Recall.
   * @param bucketAlias The alias of the bucket to query.
   * @returns A promise that resolves when all files are processed.
   */
  async retrieveAndProcessKnowledgeFiles(bucketAlias: string): Promise<void> {
    try {
      const bucketAddress = await this.getOrCreateBucket(bucketAlias);
      elizaLogger.info(
        `[RecallService] Retrieving knowledge files from bucket: ${bucketAddress}`
      );

      // Query for Parquet files
      const result = await this.client
        .bucketManager()
        .query(bucketAddress, { prefix: this.prefix });
      const queryResult: BucketQueryResponse = {
        result: {
          objects: result.result.objects.map((obj) => ({
            key: obj.key,
            size: Number(obj.state.size),
            lastModified: new Date().toISOString(), // or get from metadata if available
          })),
        },
      };

      if (!queryResult.result?.objects.length) {
        elizaLogger.info(
          `[RecallService] No knowledge files found in bucket: ${bucketAlias}`
        );
        return;
      }

      // Filter for unprocessed Parquet files
      const unprocessedFiles = queryResult.result.objects
        .map((obj) => obj.key)
        .filter((key) => key.endsWith(".parquet"))
        .filter((key) => !this.processedFiles.has(key));

      elizaLogger.info(
        `[RecallService] Found ${unprocessedFiles.length} unprocessed files out of ${
          queryResult.result.objects.length
        } total files`
      );

      // Process only new files
      for (const knowledgeFile of unprocessedFiles) {
        try {
          await this.processKnowledgeFile(bucketAddress, knowledgeFile);
        } catch (fileError) {
          // Continue with next file even if one fails
          elizaLogger.error(
            `[RecallService] Failed to process file ${knowledgeFile}, continuing with next file`
          );
        }
      }
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error retrieving knowledge files: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
    }
  }

  /**
   * Convert various data formats to Buffer
   * @param data Data in various formats
   * @returns Buffer representation of the data
   */
  private convertToBuffer(
    data: Buffer | Uint8Array | number[] | string
  ): Buffer {
    if (Buffer.isBuffer(data)) {
      return data;
    } else if (Array.isArray(data) || data instanceof Uint8Array) {
      return Buffer.from(data);
    } else if (typeof data === "object") {
      return Buffer.from(Object.values(data) as number[]);
    } else {
      throw new Error("Invalid data format for conversion to Buffer");
    }
  }

  /**
   * Process a Parquet buffer and store its contents in DuckDB
   * @param parquetBuffer Buffer containing Parquet data
   * @param knowledgeFile Filename/key associated with this buffer
   */
  private async processParquetBuffer(
    parquetBuffer: Buffer,
    knowledgeFile: string
  ): Promise<void> {
    const reader = await ParquetReader.openBuffer(parquetBuffer);
    const cursor = reader.getCursor();
    let recordCount = 0;
    let decryptedCount = 0;
    let record: Record<string, AnyType>;

    try {
      while ((record = (await cursor.next()) as Record<string, AnyType>)) {
        recordCount++;

        // Debug first record structure
        if (recordCount === 1) {
          elizaLogger.debug(
            `[RecallService] 🔍 First record structure: ${Object.keys(record).join(", ")}`
          );
        }

        const userId = record.userId;
        const agentId = record.agentId;
        const embedding = record.embedding;

        // Skip records missing critical fields
        if (!userId || !agentId || !embedding || !Array.isArray(embedding)) {
          elizaLogger.warn(
            `[RecallService] ⚠️ Skipping record missing critical fields in ${knowledgeFile}`,
            {
              recordKeys: Object.keys(record),
              hasUserId: !!userId,
              hasAgentId: !!agentId,
              hasEmbedding: !!embedding && Array.isArray(embedding),
            }
          );
          continue;
        }

        try {
          // Extract and decrypt content
          const originalText =
            record.userMessage || record.log || "No content available";

          // Decrypt content if needed and access control service is available
          const decryptedText = await this.decryptContentIfNeeded(
            originalText,
            this.accessControlService
          );
          if (decryptedText !== originalText) decryptedCount++;

          // Create Content object
          const content: Content = { text: decryptedText };

          // Generate unique memory ID
          const memoryId = stringToUuid(
            `${userId}-${agentId}-${record.timestamp || Date.now()}`
          );

          // Generate Room ID
          const roomId = stringToUuid(`room-${userId}-${agentId}`);

          // Parse timestamp
          const timestamp = record.timestamp
            ? new Date(record.timestamp).getTime()
            : Date.now();

          // Convert record to Memory format
          const memoryRecord: Memory = {
            id: memoryId,
            userId,
            agentId,
            content,
            embedding,
            roomId,
            createdAt: timestamp,
          };

          // Store in DuckDB
          await this.insertKnowledgeIntoDuckDB(memoryRecord, knowledgeFile);
        } catch (recordError) {
          elizaLogger.error(
            `[RecallService] ❌ Error processing record in ${knowledgeFile}: ${recordError.message}`,
            {
              error: recordError,
              stack: recordError.stack,
            }
          );
          continue;
        }
      }

      await reader.close();
      elizaLogger.info(
        `[RecallService] ✅ Successfully processed ${recordCount} records from ${knowledgeFile}. 🔓 Decrypted ${decryptedCount} records.`
      );
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error processing parquet buffer: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
      throw error;
    }
  }

  /**
   * Retrieves and processes a specific knowledge file from Recall.
   * @param bucketAddress The bucket address containing the file.
   * @param knowledgeFile The filename/key to process.
   * @returns A promise that resolves when the file is processed.
   */
  private async processKnowledgeFile(
    bucketAddress: Address,
    knowledgeFile: string
  ): Promise<void> {
    try {
      elizaLogger.info(
        `[RecallService] 📂 Processing knowledge file: ${knowledgeFile}`
      );

      // Retrieve the Parquet file from Recall
      const fileData = await this.client
        .bucketManager()
        .get(bucketAddress, knowledgeFile);
      if (!fileData.result) {
        elizaLogger.warn(
          `[RecallService] ⚠️ No data found in knowledge file: ${knowledgeFile}`
        );
        return;
      }

      // Convert file data to Buffer
      const parquetBuffer = this.convertToBuffer(fileData.result);

      elizaLogger.info(
        `[RecallService] ✅ Retrieved Parquet buffer (${parquetBuffer.length} bytes) for processing.`
      );

      await this.processParquetBuffer(parquetBuffer, knowledgeFile);

      await this.markFileAsProcessed(knowledgeFile);
    } catch (error) {
      elizaLogger.error(
        `[RecallService] ❌ Error processing file ${knowledgeFile}: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
      throw error;
    }
  }

  /**
   * Decrypts a message if it's encrypted; otherwise, returns it as-is.
   * @param contentText The text content of a memory entry.
   * @param accessControlService The AccessControlService instance.
   * @returns Decrypted content if encrypted, otherwise the original content.
   */
  private async decryptContentIfNeeded(
    contentText: string,
    accessControlService: AccessControlService | undefined
  ): Promise<string> {
    try {
      if (!contentText || !accessControlService) return contentText;

      // Attempt to parse JSON (to check if it's encrypted)
      const contentJson = JSON.parse(contentText);
      if (contentJson.ciphertext && contentJson.dataToEncryptHash) {
        elizaLogger.debug(
          `[RecallService] 🔓 Attempting to decrypt message...`
        );
        const decryptedText = await accessControlService.decryptMessage(
          contentJson.ciphertext,
          contentJson.dataToEncryptHash
        );
        if (decryptedText) {
          elizaLogger.info(
            `[RecallService] ✅ Successfully decrypted message.`
          );
          return decryptedText;
        }
      }
    } catch (e) {
      // Not encrypted JSON, just return original content
    }
    return contentText;
  }

  /**
   * Provides knowledge relevant to a given message for the agent's context.
   * This method can be used as a provider in the agent runtime.
   * @param message The current message being processed.
   * @returns A formatted string of relevant knowledge.
   */
  public async provideKnowledge(message: Memory): Promise<string> {
    try {
      // First, ensure we have processed any new knowledge files
      await this.retrieveAndProcessKnowledgeFiles(this.alias);

      if (!message.content.text || message.content.text.trim().length === 0) {
        return "";
      }

      // Query for relevant knowledge
      const relevantKnowledge = await this.queryKnowledge(message);

      if (!relevantKnowledge || relevantKnowledge.length === 0) {
        elizaLogger.info(
          "[RecallService] No relevant knowledge found for the message."
        );
        return "";
      }

      // Format the knowledge for the context
      const formattedKnowledge =
        this.formatKnowledgeForContext(relevantKnowledge);

      return formattedKnowledge;
    } catch (error) {
      elizaLogger.error(
        `[RecallService] Error providing knowledge: ${error.message}`
      );
      return "";
    }
  }
}
