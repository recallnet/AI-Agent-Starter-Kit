import {
  elizaLogger,
  type UUID,
  Service,
  ServiceType,
  stringToUuid,
  IAgentRuntime,
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
import { AnyType } from "src/utils.js";
import { Content } from "@ai16z/eliza";
import { randomUUID } from "crypto";

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

// Load environment variables with detailed logging
const privateKey = process.env.RECALL_PRIVATE_KEY as Hex;
const envAlias = process.env.RECALL_BUCKET_ALIAS as string;
const envPrefix = process.env.RECALL_MEMORY_PREFIX as string;
const network = process.env.RECALL_NETWORK as string;

// Add debug logging for environment variables
elizaLogger.info("Environment configuration:", {
  RECALL_PRIVATE_KEY: privateKey ? "[REDACTED]" : undefined,
  RECALL_BUCKET_ALIAS: envAlias,
  RECALL_MEMORY_PREFIX: envPrefix,
  RECALL_NETWORK: network,
});

export class RecallService extends Service {
  static get serviceType(): ServiceType {
    elizaLogger.info("Getting RecallService.serviceType");
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

  getInstance(): RecallService {
    elizaLogger.info("RecallService.getInstance() called");
    return this;
  }

  constructor(_runtime: IAgentRuntime) {
    super();
    elizaLogger.info("RecallService constructor called");
    try {
      elizaLogger.info("RecallService super() constructor completed");
      this.runtime = _runtime;
      elizaLogger.info("RecallService constructor runtime assigned", {
        runtimeExists: !!_runtime,
        runtimeType: _runtime ? typeof _runtime : "undefined",
      });
    } catch (error) {
      elizaLogger.error(
        `Error in RecallService constructor: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
      throw error;
    }
  }

  async initialize(runtime: IAgentRuntime): Promise<void> {
    elizaLogger.info("RecallService.initialize() called", {
      hasRuntime: !!runtime,
      hasThisRuntime: !!this.runtime,
    });

    try {
      // Guard against multiple initializations
      if (this.isInitialized) {
        elizaLogger.warn("RecallService already initialized, skipping");
        return;
      }

      // Validate environment variables
      if (!privateKey) {
        elizaLogger.error("RECALL_PRIVATE_KEY is required");
        throw new Error("RECALL_PRIVATE_KEY is required");
      }
      if (!envAlias) {
        elizaLogger.error("RECALL_BUCKET_ALIAS is required");
        throw new Error("RECALL_BUCKET_ALIAS is required");
      }
      if (!envPrefix) {
        elizaLogger.error("RECALL_MEMORY_PREFIX is required");
        throw new Error("RECALL_MEMORY_PREFIX is required");
      }

      // Use runtime from parameter if provided, fallback to constructor runtime
      if (runtime) {
        elizaLogger.info("Using runtime from initialize() parameter");
        this.runtime = runtime;
      } else if (!this.runtime) {
        elizaLogger.error("No runtime available for initialization");
        throw new Error("No runtime available for initialization");
      }

      elizaLogger.info("RecallService initialization started");

      // Set up blockchain connection
      elizaLogger.info(
        `Setting up blockchain connection with network: ${network || "testnet"}`
      );
      const chain = network ? getChain(network as ChainName) : testnet;
      elizaLogger.info("Creating wallet client from private key");
      const wallet = walletClientFromPrivateKey(privateKey, chain);
      elizaLogger.info("Creating RecallClient");
      this.client = new RecallClient({ walletClient: wallet });

      // Set configuration values
      this.alias = envAlias;
      this.prefix = envPrefix;
      elizaLogger.info(
        `RecallService configured with alias: ${this.alias}, prefix: ${this.prefix}`
      );

      // Initialize DuckDB
      elizaLogger.info("Initializing DuckDB in-memory database");
      try {
        const db = new duckdb.Database(":memory:"); // In-memory DB for performance
        this.db = db.connect();
        elizaLogger.info("DuckDB connection established");
      } catch (dbError) {
        elizaLogger.error(`Failed to initialize DuckDB: ${dbError.message}`, {
          error: dbError,
          stack: dbError.stack,
        });
        throw dbError;
      }

      // Create database schema
      elizaLogger.info("Creating DuckDB schema");
      try {
        await new Promise<void>((resolve, reject) => {
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
                elizaLogger.error(`⛔ Error creating schema: ${err.message}`, {
                  error: err,
                  stack: err.stack,
                });
                reject(err);
              } else {
                elizaLogger.info("✅ Schema created successfully");
                resolve();
              }
            }
          );
        });

        elizaLogger.info("DuckDB schema creation completed");
      } catch (schemaError) {
        elizaLogger.error(`Failed to create schema: ${schemaError.message}`);
        throw schemaError;
      }

      // Load processed files into memory
      elizaLogger.info("Loading processed files into memory");
      try {
        await this.loadProcessedFiles();
        elizaLogger.info(`Loaded ${this.processedFiles.size} processed files`);
      } catch (loadError) {
        elizaLogger.error(
          `Error loading processed files: ${loadError.message}`
        );
        throw loadError;
      }

      // Initialize accessControlService
      await this.initializeAccessControlService();
      elizaLogger.info("accessControlService initialized");

      this.isInitialized = true;
      elizaLogger.success("RecallService initialized successfully");
    } catch (error) {
      elizaLogger.error(`Error initializing RecallService: ${error.message}`, {
        error,
        stack: error.stack,
        runtimeExists: !!this.runtime,
      });
      throw error;
    }
  }

  /**
   * Initialize the AccessControlService during RecallService initialization
   * Add this to your initialize() method
   */
  async initializeAccessControlService(): Promise<boolean> {
    try {
      elizaLogger.info(
        "Initializing AccessControlService for encryption/decryption"
      );
      this.accessControlService = AccessControlService.getInstance();
      await this.accessControlService.start();

      if (!this.accessControlService.isConfigured()) {
        elizaLogger.warn(
          "AccessControlService is not properly configured. Content will not be encrypted."
        );
        return false;
      } else {
        elizaLogger.info(
          "AccessControlService initialized successfully. Content encryption is enabled."
        );
        return true;
      }
    } catch (error) {
      elizaLogger.error(
        `Error initializing AccessControlService: ${error.message}`
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
   * Loads processed files from DuckDB into memory.
   * @returns A promise that resolves when the files are loaded.
   */
  private async loadProcessedFiles(): Promise<void> {
    elizaLogger.info("Loading processed files from DuckDB");
    return new Promise((resolve, reject) => {
      // The SQL query doesn't need any parameters, so provide an empty array
      this.db.all(
        "SELECT fileKey FROM processed_files;",
        (err: Error | null, rows: AnyType) => {
          if (err) {
            elizaLogger.error(
              `Error querying processed files: ${err.message}`,
              {
                error: err,
                stack: err.stack,
              }
            );
            reject(err);
            return;
          }
          try {
            this.processedFiles = new Set(
              rows.map((row: { fileKey: string }) => row.fileKey)
            );
            elizaLogger.info(
              `Loaded ${this.processedFiles.size} processed files from database`
            );
            resolve();
          } catch (mapError) {
            elizaLogger.error(
              `Error processing query results: ${mapError.message}`,
              {
                error: mapError,
                stack: mapError.stack,
                rows: rows ? `${rows.length} rows` : "undefined",
              }
            );
            reject(mapError);
          }
        }
      );
    });
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

    await new Promise<void>((resolve, reject) => {
      this.db.run(
        "INSERT INTO processed_files (fileKey, processedAt) VALUES (?, ?);",
        fileKey,
        new Date().toISOString(),
        (err) => {
          if (err) reject(err);
          else {
            this.processedFiles.add(fileKey);
            resolve();
          }
        }
      );
    });
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
      elizaLogger.error(`Error getting account info: ${error.message}`);
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
      elizaLogger.error(`Error listing buckets: ${error.message}`);
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
      elizaLogger.error(`Error getting credit info: ${error.message}`);
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
      elizaLogger.error(`Error buying credit: ${error.message}`);
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
      elizaLogger.info(`Looking for bucket with alias: ${bucketAlias}`);

      // Try to find the bucket by alias
      const buckets = await this.client.bucketManager().list();
      if (buckets?.result) {
        const bucket = buckets.result.find(
          (b) => b.metadata?.alias === bucketAlias
        );
        if (bucket) {
          elizaLogger.info(
            `Found existing bucket "${bucketAlias}" at ${bucket.addr}`
          );
          return bucket.addr; // Return existing bucket address
        } else {
          elizaLogger.info(
            `Bucket with alias "${bucketAlias}" not found, creating a new one.`
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
          `Failed to create new bucket with alias: ${bucketAlias}`
        );
        throw new Error(`Failed to create bucket: ${bucketAlias}`);
      }

      elizaLogger.info(
        `Successfully created new bucket "${bucketAlias}" at ${newBucket.bucket}`
      );
      return newBucket.bucket;
    } catch (error) {
      elizaLogger.error(`Error in getOrCreateBucket: ${error.message}`);
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
      elizaLogger.error(`Error adding object: ${error.message}`);
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
      elizaLogger.warn(`Error getting object: ${error.message}`);
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
        `Validation failed at ${context}: memory is null or undefined`
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
          memory.embedding = JSON.parse(memory.embedding);
        } catch (e) {
          elizaLogger.error(
            `Failed to parse embedding string at ${context}: ${e.message}`
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
          `Embedding validation failed at ${context} for memory ${validation.memoryId}`,
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
        `Error during embedding validation at ${context}: ${error.message}`
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
              `Using existing embedding for memory ${memory.id}`,
              {
                embeddingLength: preparedMemory.embedding.length,
              }
            );
            continue;
          } else {
            elizaLogger.warn(
              `Failed to copy embedding for memory ${memory.id}, will try to generate new one`
            );
          }
        }

        // Need to generate new embedding
        elizaLogger.info(`Generating new embedding for memory ${memory.id}`);

        // Create a clean copy without any existing embedding
        const memoryForEmbedding = {
          ...memory,
          embedding: undefined,
          id: memory.id || stringToUuid(randomUUID()),
        };

        // Try to generate embedding with retries
        let attemptsLeft = 3;
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
                `Invalid embedding generated for memory ${memory.id}, ${attemptsLeft} attempts remaining`
              );
              await new Promise((resolve) => setTimeout(resolve, 1000));
            }
          } catch (embedError) {
            attemptsLeft--;
            if (attemptsLeft > 0) {
              elizaLogger.warn(
                `Error generating embedding for memory ${memory.id}, ${attemptsLeft} attempts remaining: ${embedError.message}`
              );
              await new Promise((resolve) => setTimeout(resolve, 1000));
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
            `Successfully added new embedding to memory ${memory.id}`,
            {
              embeddingLength: generatedMemory.embedding?.length,
            }
          );
        } else {
          throw new Error(
            `Failed to generate valid embedding for memory ${memory.id} after all attempts`
          );
        }
      } catch (error) {
        elizaLogger.error(
          `Failed to prepare memory ${memory.id}: ${error.message}`,
          {
            error,
            stack: error.stack,
            memoryContent:
              typeof memory.content === "string"
                ? "string content"
                : memory.content,
          }
        );
      }
    }

    // Log overall results
    elizaLogger.info(`Memory preparation complete`, {
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
        `${invalidMemories.length} memories have invalid embeddings after preparation`
      );
      elizaLogger.error(error.message, {
        invalidMemoryIds: invalidMemories.map((m) => m.id),
      });
      throw error;
    }

    return preparedMemories;
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
      elizaLogger.info(`Preparing ${batch.length} memories for storage`);
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
          `${missingEmbeddings.length} memories still missing embeddings after preparation`,
          {
            memoryIds: missingEmbeddings.map((m) => m.id),
          }
        );
        return undefined;
      }

      if (preparedMemories.length === 0) {
        elizaLogger.warn("No valid memories to store after preparation.");
        return undefined;
      }

      elizaLogger.info(
        `Transforming ${preparedMemories.length} memories to knowledge format`
      );

      // Encrypt memory content before storing
      const encryptedMemories = await Promise.all(
        preparedMemories.map(async (memory) => {
          let finalMemory;
          if (!this.accessControlService) {
            elizaLogger.warn(
              "AccessControlService is unavailable, storing data without encryption."
            );
            finalMemory = memory;
          } else {
            finalMemory =
              await this.accessControlService.prepareMemoryForStorage(memory);
            if (!finalMemory) {
              elizaLogger.error(`Failed to encrypt memory ${memory.id}`);
              return memory;
            }
          }
          return finalMemory;
        })
      );

      const memoryRecords = encryptedMemories.map((memory) => {
        // Log the memory structure before transformation
        elizaLogger.debug(`Memory pre-transform:`, {
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
          `Creating record from memory ID=${memory.id}, roomId=${memory.roomId}`
        );

        if (!memory.embedding || !Array.isArray(memory.embedding)) {
          throw new Error(
            `Memory ${memory.id} has invalid embedding during transformation`
          );
        }

        const record = {
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
        elizaLogger.debug(`Knowledge record created:`, {
          userId: record.userId,
          hasEmbedding: Array.isArray(record.embedding),
          embeddingLength: record.embedding?.length,
          recordKeys: Object.keys(record),
        });

        return record;
      });

      // Verify all records have valid embeddings before proceeding
      const withoutEmbeddings = memoryRecords.filter(
        (record) =>
          !record.embedding ||
          !Array.isArray(record.embedding) ||
          record.embedding.length === 0
      );

      if (withoutEmbeddings.length > 0) {
        elizaLogger.error(
          `${withoutEmbeddings.length} records missing embeddings before Parquet creation`,
          {
            records: withoutEmbeddings.map((r) => ({
              userId: r.userId,
              hasEmbedding: !!r.embedding,
              embeddingLength: r.embedding?.length,
            })),
          }
        );
        return undefined;
      }

      // More detailed logging around Parquet creation
      elizaLogger.info(
        `Attempting to create Parquet buffer for ${memoryRecords.length} records with structure:`,
        {
          sampleKeys: Object.keys(memoryRecords[0]),
          hasSampleEmbedding: !!memoryRecords[0].embedding,
          sampleEmbeddingLength: memoryRecords[0].embedding?.length,
        }
      );

      try {
        // Create the Parquet schema to match the CoT format
        const parquetBuffer = await writeParquetToBuffer(memoryRecords);

        if (!parquetBuffer) {
          elizaLogger.error("writeParquetToBuffer returned undefined");
          return undefined;
        }

        if (parquetBuffer.length === 0) {
          elizaLogger.error(
            "Generated Parquet file is empty. Skipping upload."
          );
          return undefined;
        }

        elizaLogger.info(
          `Successfully generated Parquet buffer of ${parquetBuffer.length} bytes`
        );

        // Upload to Recall
        elizaLogger.info(
          `Uploading Parquet data to bucket ${bucketAddress} with key ${nextKnowledgeKey}`
        );
        const addObject = await this.withTimeout(
          this.client.bucketManager().add(
            bucketAddress,
            nextKnowledgeKey,
            Uint8Array.from(parquetBuffer) // Convert Buffer to Uint8Array
          ),
          30000,
          "Recall batch storage"
        );

        if (!addObject?.meta?.tx) {
          elizaLogger.error(
            "❌ Recall API returned invalid response for batch storage",
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
          `Successfully stored batch of ${memoryRecords.length} records at key: ${nextKnowledgeKey}`
        );
        return nextKnowledgeKey;
      } catch (parquetError) {
        elizaLogger.error(
          `Parquet generation/upload error: ${parquetError.message}`,
          {
            error: parquetError,
            stack: parquetError.stack,
          }
        );
        return undefined;
      }
    } catch (error) {
      elizaLogger.error(
        `Error storing knowledge as Parquet in Recall: ${error.message}`,
        {
          error,
          stack: error.stack,
        }
      );
      return undefined;
    }
  }

  async writeKnowledgeToRecall(message: Memory): Promise<string | undefined> {
    try {
      const bucketAddress = await this.withTimeout(
        this.getOrCreateBucket(this.alias),
        15000,
        "Get/Create bucket"
      );

      elizaLogger.info(`📂 Processing message ID=${message.id}`);

      const knowledgeFileKey = await this.storeBatchToRecall(
        bucketAddress,
        [message] // Syncing only the input message
      );

      if (knowledgeFileKey) {
        await this.insertKnowledgeIntoDuckDB(message, knowledgeFileKey);
        elizaLogger.success(`✅ Successfully synced message ID=${message.id}`);
        return knowledgeFileKey;
      } else {
        elizaLogger.warn(
          `⚠️ Failed to sync message ID=${message.id} - will retry on next sync`
        );
        return undefined;
      }
    } catch (error) {
      if (error.message.includes("timed out")) {
        elizaLogger.error(
          `⏳ Recall sync operation timed out: ${error.message}`
        );
        return undefined;
      } else {
        elizaLogger.error(
          `❌ Error in writeKnowledgeToRecall: ${error.message}`
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
        `Memory ${memory.id} has no embedding, skipping insertion into DuckDB`
      );
      return;
    }

    // Debug the values we're about to insert
    elizaLogger.debug(
      `Preparing to insert memory into DuckDB: ID=${memory.id}, file=${knowledgeFileKey}`
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
      await new Promise<void>((resolve, reject) => {
        // Debug all parameters to ensure we have the correct count
        const params = [
          memoryId,
          memory.userId,
          memory.agentId,
          contentStr,
          JSON.stringify(embeddingArray),
          memory.roomId,
          createdAtStr,
          knowledgeFileKey,
        ];

        elizaLogger.debug(
          `SQL insert parameters: ${params.length} parameters`,
          {
            paramCount: params.length,
          }
        );

        this.db.run(
          `INSERT INTO knowledge 
       VALUES (?, ?, ?, ?, CAST(? AS FLOAT[]), ?, ?, ?)
       ON CONFLICT (id, knowledgeFileKey) DO NOTHING;`,
          ...params,
          (err) => {
            if (err) {
              elizaLogger.error(`Error in SQL insert: ${err.message}`, {
                error: err,
                params: params.map(
                  (p, i) =>
                    `param${i}: ${typeof p} ${p === null ? "null" : typeof p === "string" ? p.substring(0, 50) + "..." : p}`
                ),
              });
              reject(err);
            } else {
              elizaLogger.debug(
                `Successfully inserted memory ${memoryId} into DuckDB`
              );
              resolve();
            }
          }
        );
      });
    } catch (error) {
      if (!error.message.includes("UNIQUE constraint")) {
        elizaLogger.error(`Error inserting knowledge: ${error.message}`, {
          error,
          stack: error.stack,
        });
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
  async queryKnowledge(
    queryText: Memory,
    limit = 10,
    threshold = 0.7
  ): Promise<KnowledgeItem[]> {
    try {
      elizaLogger.info("📡 queryKnowledge() called", {
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
        elizaLogger.error("❌ Failed to generate embedding for query text.");
        return [];
      }

      elizaLogger.info("🔍 Query Embedding Generated", {
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
        ORDER BY similarity DESC
        LIMIT ${limit};
      `;

      elizaLogger.info("📝 Executing SQL Query:", { query });

      // Execute SQL query in DuckDB
      const searchResults: AnyType[] = await new Promise((resolve, reject) => {
        this.db.all(query, (err, res) => {
          if (err) {
            elizaLogger.error("⛔ Error executing DuckDB query!", {
              error: err.message,
              stack: err.stack,
            });
            reject(err);
          } else {
            elizaLogger.info(
              `✅ Query executed successfully. Found ${res.length} results.`
            );
            resolve(res || []);
          }
        });
      });

      // Check if any results were found
      if (!searchResults || searchResults.length === 0) {
        elizaLogger.warn("⚠️ No similar knowledge found.");
        return [];
      }

      // Transform results into KnowledgeItem format
      return searchResults.map((result) => {
        let contentObj: AnyType;

        try {
          contentObj =
            typeof result.content === "string"
              ? JSON.parse(result.content)
              : result.content;

          if (!contentObj.text && typeof contentObj === "string") {
            contentObj = { text: contentObj };
          }
        } catch (e) {
          contentObj = { text: result.content };
        }

        return {
          id: result.id,
          content: contentObj,
          similarity: parseFloat(result.similarity),
        };
      });
    } catch (error) {
      elizaLogger.error("❌ Error in queryKnowledge:", { error });
      return [];
    }
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
        `Retrieving knowledge files from bucket: ${bucketAddress}`
      );

      // Query for Parquet files
      const queryResult = await this.client
        .bucketManager()
        .query(bucketAddress, { prefix: this.prefix });

      if (!queryResult.result?.objects.length) {
        elizaLogger.info(`No knowledge files found in bucket: ${bucketAlias}`);
        return;
      }

      // Filter for unprocessed Parquet files
      const unprocessedFiles = queryResult.result.objects
        .map((obj) => obj.key)
        .filter((key) => key.endsWith(".parquet"))
        .filter((key) => !this.processedFiles.has(key));

      elizaLogger.info(
        `Found ${unprocessedFiles.length} unprocessed files out of ${
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
            `Failed to process file ${knowledgeFile}, continuing with next file`
          );
        }
      }
    } catch (error) {
      elizaLogger.error(`Error retrieving knowledge files: ${error.message}`, {
        error,
        stack: error.stack,
      });
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
      elizaLogger.info(`📂 Processing knowledge file: ${knowledgeFile}`);

      // Retrieve the Parquet file from Recall
      const fileData = await this.client
        .bucketManager()
        .get(bucketAddress, knowledgeFile);
      if (!fileData.result) {
        elizaLogger.warn(
          `⚠️ No data found in knowledge file: ${knowledgeFile}`
        );
        return;
      }

      // Convert file data to Buffer
      let parquetBuffer: Buffer;
      if (Buffer.isBuffer(fileData.result)) {
        parquetBuffer = fileData.result;
      } else if (
        Array.isArray(fileData.result) ||
        fileData.result instanceof Uint8Array
      ) {
        parquetBuffer = Buffer.from(fileData.result);
      } else if (typeof fileData.result === "object") {
        parquetBuffer = Buffer.from(Object.values(fileData.result) as number[]);
      } else {
        throw new Error(
          `Invalid fileData.result format for knowledge file: ${knowledgeFile}`
        );
      }

      elizaLogger.info(
        `✅ Retrieved Parquet buffer (${parquetBuffer.length} bytes) for processing.`
      );

      const reader = await ParquetReader.openBuffer(parquetBuffer);
      const cursor = reader.getCursor();
      let recordCount = 0;
      let decryptedCount = 0;
      let record: AnyType;

      while ((record = await cursor.next())) {
        recordCount++;

        // Debug first record structure
        if (recordCount === 1) {
          elizaLogger.debug(
            `🔍 First record structure: ${Object.keys(record).join(", ")}`
          );
        }

        const userId = record.userId;
        const agentId = record.agentId;
        const embedding = record.embedding;

        // Skip records missing critical fields
        if (!userId || !agentId || !embedding || !Array.isArray(embedding)) {
          elizaLogger.warn(
            `⚠️ Skipping record missing critical fields in ${knowledgeFile}`,
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
          if (!this.accessControlService) {
            return originalText;
          }
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
            `❌ Error processing record in ${knowledgeFile}: ${recordError.message}`,
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
        `✅ Successfully processed ${recordCount} records from ${knowledgeFile}. 🔓 Decrypted ${decryptedCount} records.`
      );
      await this.markFileAsProcessed(knowledgeFile);
    } catch (error) {
      elizaLogger.error(
        `❌ Error processing file ${knowledgeFile}: ${error.message}`,
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
    accessControlService: AccessControlService
  ): Promise<string> {
    try {
      if (!contentText) return contentText;

      // Attempt to parse JSON (to check if it's encrypted)
      const contentJson = JSON.parse(contentText);
      if (contentJson.ciphertext && contentJson.dataToEncryptHash) {
        elizaLogger.debug(`🔓 Attempting to decrypt message...`);
        const decryptedText = await accessControlService.decryptMessage(
          contentJson.ciphertext,
          contentJson.dataToEncryptHash
        );
        if (decryptedText) {
          elizaLogger.info(`✅ Successfully decrypted message.`);
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
        elizaLogger.info("No relevant knowledge found for the message.");
        return "";
      }

      // Format the knowledge for the context
      const formattedKnowledge =
        this.formatKnowledgeForContext(relevantKnowledge);

      return formattedKnowledge;
    } catch (error) {
      elizaLogger.error(`Error providing knowledge: ${error.message}`);
      return "";
    }
  }
}
