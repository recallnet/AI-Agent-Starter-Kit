import {
  elizaLogger,
  type UUID,
  Service,
  ServiceType,
  stringToUuid,
  IAgentRuntime,
} from "@ai16z/eliza";
import duckdb from "duckdb";
import { TextEncoder } from "util";
import { ParquetReader } from "@dsnp/parquetjs";
import { writeParquetToBuffer } from "./stream.service.js";
import { ChainName, getChain, testnet } from "@recallnet/chains";
import { AccountInfo } from "@recallnet/sdk/account";
import { ListResult } from "@recallnet/sdk/bucket";
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
const envPrefix = process.env.RECALL_COT_LOG_PREFIX as string;
const network = process.env.RECALL_NETWORK as string;
const intervalPeriod = process.env.RECALL_SYNC_INTERVAL as string;
const batchSize = process.env.RECALL_BATCH_SIZE as string;

// Add debug logging for environment variables
elizaLogger.info("Environment configuration:", {
  RECALL_PRIVATE_KEY: privateKey ? "[REDACTED]" : undefined,
  RECALL_BUCKET_ALIAS: envAlias,
  RECALL_COT_LOG_PREFIX: envPrefix,
  RECALL_NETWORK: network,
  RECALL_SYNC_INTERVAL: intervalPeriod,
  RECALL_BATCH_SIZE: batchSize,
});

export class RecallService extends Service {
  static get serviceType(): ServiceType {
    elizaLogger.info("Getting RecallService.serviceType");
    return "recall" as ServiceType;
  }

  private client: RecallClient;
  private runtime: IAgentRuntime;
  private syncInterval: NodeJS.Timeout | undefined;
  private alias: string;
  private prefix: string;
  private db: duckdb.Connection;
  private processedFiles: Set<string> = new Set();
  private intervalMs: number;
  private batchSizeKB: number;
  private lastSyncTime: number = 0;
  private isInitialized: boolean = false;

  getInstance(): RecallService {
    elizaLogger.info("RecallService.getInstance() called");
    return RecallService.getInstance();
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
        elizaLogger.error("RECALL_COT_LOG_PREFIX is required");
        throw new Error("RECALL_COT_LOG_PREFIX is required");
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
          this.db.run(
            `CREATE TABLE knowledge (
                id TEXT,
                userId TEXT,
                agentId TEXT,
                content TEXT,
                embedding FLOAT[], 
                roomId TEXT,
                createdAt TEXT,
                knowledgeFileKey TEXT,
                UNIQUE(id, knowledgeFileKey)
            );
            
            CREATE TABLE processed_files (
                fileKey TEXT PRIMARY KEY,
                processedAt TEXT
            );`,
            (err) => {
              if (err) {
                elizaLogger.error(`Error creating schema: ${err.message}`, {
                  error: err,
                  stack: err.stack,
                });
                reject(err);
              } else {
                elizaLogger.info("Schema created successfully");
                resolve();
              }
            }
          );
        });
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

      // Set up sync configuration
      elizaLogger.info("Setting up sync configuration");
      this.intervalMs = intervalPeriod
        ? parseInt(intervalPeriod, 10)
        : 2 * 60 * 1000;
      this.batchSizeKB = batchSize ? parseInt(batchSize, 10) : 4;
      elizaLogger.info(
        `Sync configuration: interval=${this.intervalMs}ms, batchSize=${this.batchSizeKB}KB`
      );

      // Start periodic sync
      elizaLogger.info("Starting periodic sync");
      try {
        this.startPeriodicSync(this.intervalMs, this.batchSizeKB);
        elizaLogger.info("Periodic sync started successfully");
      } catch (syncError) {
        elizaLogger.error(`Error starting periodic sync: ${syncError.message}`);
        throw syncError;
      }

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
   * Prepares memories for storage by ensuring they have embeddings.
   * @param memories Array of memories to prepare
   * @returns Array of prepared memories with embeddings
   */
  private async prepareMemoriesForStorage(
    memories: Memory[]
  ): Promise<Memory[]> {
    const preparedMemories: Memory[] = [];

    for (const memory of memories) {
      // Skip if already has embedding
      if (memory.embedding && memory.embedding.length > 0) {
        preparedMemories.push(memory);
        continue;
      }

      try {
        // Add embedding if missing
        const memoryWithEmbedding =
          await this.runtime.messageManager.addEmbeddingToMemory(memory);
        preparedMemories.push(memoryWithEmbedding);
      } catch (error) {
        elizaLogger.error(`Error adding embedding to memory: ${error.message}`);
      }
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
      const preparedMemories = await this.prepareMemoriesForStorage(batch);
      if (preparedMemories.length === 0) {
        elizaLogger.warn("No valid memories to store after preparation.");
        return undefined;
      }

      // Transform to the expected format for Chain of Thought logs
      // This matches the format we read from existing Parquet files
      elizaLogger.info(
        `Transforming ${preparedMemories.length} memories to Chain of Thought format`
      );
      const cotRecords = preparedMemories.map((memory) => {
        // Extract text from content object or handle string content
        const text =
          typeof memory.content === "string"
            ? memory.content
            : memory.content.text || "";

        // Log record structure for debugging
        elizaLogger.debug(
          `Creating COT record from memory ID=${memory.id}, roomId=${memory.roomId}`
        );

        // Create a record matching the Chain of Thought format
        return {
          userId: memory.userId || "",
          agentId: memory.agentId || "",
          userMessage: text, // Store content as userMessage
          log: text, // Also store in log for redundancy
          embedding: Array.isArray(memory.embedding) ? memory.embedding : [],
          timestamp: memory.createdAt
            ? new Date(memory.createdAt).toISOString()
            : new Date().toISOString(),
        };
      });

      if (cotRecords.length === 0) {
        elizaLogger.warn("No valid Chain of Thought records to store.");
        return undefined;
      }

      // More detailed logging around Parquet creation
      elizaLogger.info(
        `Attempting to create Parquet buffer for ${cotRecords.length} records`
      );

      try {
        // Make sure the writeParquetToBuffer function is properly imported
        if (typeof writeParquetToBuffer !== "function") {
          elizaLogger.error("writeParquetToBuffer is not a function", {
            type: typeof writeParquetToBuffer,
          });
          return undefined;
        }

        // Create the Parquet buffer - make sure it matches the expected format
        const parquetBuffer = await writeParquetToBuffer(cotRecords);

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
          this.client
            .bucketManager()
            .add(bucketAddress, nextKnowledgeKey, parquetBuffer),
          30000, // 30-second timeout
          "Recall batch storage"
        );

        if (!addObject?.meta?.tx) {
          elizaLogger.error(
            "Recall API returned invalid response for batch storage",
            { response: JSON.stringify(addObject) }
          );
          return undefined;
        }

        elizaLogger.info(
          `Successfully stored batch of ${cotRecords.length} records at key: ${nextKnowledgeKey}`
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
        { error, stack: error.stack }
      );
      return undefined;
    }
  }

  /**
   * Syncs knowledge to Recall in batches.
   * @param bucketAlias The alias of the bucket to store knowledge.
   * @param batchSizeKB The maximum size of each batch in kilobytes.
   */
  async syncKnowledgeToRecall(
    bucketAlias: string,
    batchSizeKB = 4
  ): Promise<void> {
    try {
      const currentTime = Date.now();

      // Get bucket address
      const bucketAddress = await this.withTimeout(
        this.getOrCreateBucket(bucketAlias),
        15000,
        "Get/Create bucket"
      );

      // Since we don't have direct access to all rooms, we'll use a different approach:
      // Get unsynced memories since the last sync time
      const unsyncedMemories = await this.getUnsyncedMemories(
        this.lastSyncTime,
        currentTime
      );

      if (!unsyncedMemories || unsyncedMemories.length === 0) {
        elizaLogger.info("No new memories found for synchronization.");
        return;
      }

      elizaLogger.info(`Found ${unsyncedMemories.length} new memories to sync`);

      // Group memories by roomId for better logging and organization
      const memoriesByRoom = this.groupMemoriesByRoom(unsyncedMemories);

      let syncedCount = 0;
      const totalMemoriesFound = unsyncedMemories.length;

      // Process each room's memories
      for (const [roomId, memories] of Object.entries(memoriesByRoom)) {
        try {
          elizaLogger.info(
            `Processing ${memories.length} memories for room ${roomId}`
          );

          // Process memories in batches
          let batch: Memory[] = [];
          let batchSize = 0;

          for (const memory of memories) {
            // Serialize the memory to estimate its size
            const memoryStr = JSON.stringify(memory);
            const memorySize = new TextEncoder().encode(memoryStr).length;

            // If adding this memory would exceed the batch size, store the current batch
            if (
              batchSize + memorySize > batchSizeKB * 1024 &&
              batch.length > 0
            ) {
              elizaLogger.info(
                `Batch size limit reached (${batchSize} bytes). Uploading batch of ${batch.length} memories...`
              );

              const knowledgeFileKey = await this.storeBatchToRecall(
                bucketAddress,
                batch
              );

              if (knowledgeFileKey) {
                // Insert into local DuckDB for each memory in the batch
                for (const syncedMemory of batch) {
                  await this.insertKnowledgeIntoDuckDB(
                    syncedMemory,
                    knowledgeFileKey
                  );
                }

                syncedCount += batch.length;
                elizaLogger.info(
                  `Successfully synced batch of ${batch.length} memories`
                );
              } else {
                elizaLogger.warn(
                  `Failed to sync batch of ${batch.length} memories - will retry on next sync`
                );
              }

              // Reset batch
              batch = [];
              batchSize = 0;
            }

            // Add memory to current batch
            batch.push(memory);
            batchSize += memorySize;
          }

          // Process any remaining memories in the batch
          if (batch.length > 0) {
            elizaLogger.info(
              `Processing final batch of ${batch.length} memories (${batchSize} bytes)...`
            );

            const knowledgeFileKey = await this.storeBatchToRecall(
              bucketAddress,
              batch
            );

            if (knowledgeFileKey) {
              // Insert into local DuckDB for each memory in the batch
              for (const syncedMemory of batch) {
                await this.insertKnowledgeIntoDuckDB(
                  syncedMemory,
                  knowledgeFileKey
                );
              }

              syncedCount += batch.length;
              elizaLogger.info(
                `Successfully synced final batch of ${batch.length} memories`
              );
            } else {
              elizaLogger.warn(
                `Failed to sync final batch of ${batch.length} memories - will retry on next sync`
              );
            }
          }
        } catch (error) {
          elizaLogger.error(
            `Error processing memories for room ${roomId}: ${error.message}`
          );
        }
      }

      // Update last sync time if we successfully found and processed memories
      if (totalMemoriesFound > 0) {
        this.lastSyncTime = currentTime;
      }

      elizaLogger.info(
        `Sync cycle complete. Synced ${syncedCount}/${totalMemoriesFound} memories. Next sync in ${this.intervalMs / 1000} seconds.`
      );
    } catch (error) {
      if (error.message.includes("timed out")) {
        elizaLogger.error(`Recall sync operation timed out: ${error.message}`);
      } else {
        elizaLogger.error(`Error in syncKnowledgeToRecall: ${error.message}`);
      }
    }
  }

  /**
   * Retrieves unsynced memories within a specified time range.
   * @param startTime The start timestamp to retrieve memories from.
   * @param endTime The end timestamp to retrieve memories until.
   * @returns A Promise resolving to an array of unsynced Memory objects.
   */
  private async getUnsyncedMemories(
    startTime: number,
    endTime: number
  ): Promise<Memory[]> {
    try {
      // Get list of room IDs the agent participates in
      // We can use the databaseAdapter to get rooms where our agent is a participant
      const agentRooms =
        await this.runtime.databaseAdapter.getRoomsForParticipant(
          this.runtime.agentId
        );

      if (!agentRooms || agentRooms.length === 0) {
        elizaLogger.info("No rooms found for the agent.");
        return [];
      }

      // Collect memories from all rooms
      let allMemories: Memory[] = [];

      for (const roomId of agentRooms) {
        try {
          // Use the memory manager to get memories within the time range
          const roomMemories = await this.runtime.messageManager.getMemories({
            roomId,
            start: startTime,
            end: endTime,
          });

          if (roomMemories && roomMemories.length > 0) {
            allMemories = allMemories.concat(roomMemories);
          }
        } catch (error) {
          elizaLogger.error(
            `Error getting memories for room ${roomId}: ${error.message}`
          );
        }
      }

      // Ensure all memories have embeddings
      const memoriesWithEmbeddings =
        await this.prepareMemoriesForStorage(allMemories);

      return memoriesWithEmbeddings;
    } catch (error) {
      elizaLogger.error(`Error getting unsynced memories: ${error.message}`);
      return [];
    }
  }

  /**
   * Groups memories by their room IDs.
   * @param memories Array of memories to group.
   * @returns An object mapping room IDs to arrays of memories.
   */
  private groupMemoriesByRoom(memories: Memory[]): Record<string, Memory[]> {
    const result: Record<string, Memory[]> = {};

    for (const memory of memories) {
      if (!result[memory.roomId]) {
        result[memory.roomId] = [];
      }

      result[memory.roomId].push(memory);
    }

    return result;
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
   * Searches for knowledge in the DuckDB database using embedding similarity.
   * @param queryEmbedding The embedding to search with.
   * @param roomId The room ID to search in.
   * @param limit The maximum number of results to return.
   * @param threshold The minimum similarity threshold.
   * @returns An array of knowledge search results.
   */
  async searchKnowledgeByEmbedding(
    queryEmbedding: number[],
    roomId: string,
    limit = 5,
    threshold = 0.7
  ): Promise<KnowledgeSearchResult[]> {
    try {
      const queryEmbeddingArray = `ARRAY[${queryEmbedding.join(",")}]`;

      // Perform similarity search with filtering and thresholding
      const searchResults: AnyType[] = await new Promise((resolve, reject) => {
        this.db.all(
          `SELECT id, userId, agentId, content, roomId, createdAt,
              1 - (embedding <-> ${queryEmbeddingArray}) AS similarity 
           FROM knowledge 
           WHERE roomId = ? AND similarity > ?
           ORDER BY similarity DESC 
           LIMIT ?;`,
          roomId,
          threshold,
          limit,
          (err, res) => {
            if (err) reject(err);
            else resolve(res || []);
          }
        );
      });

      if (!searchResults || searchResults.length === 0) {
        elizaLogger.info(`No matching knowledge found for roomId ${roomId}`);
        return [];
      }

      // Map results to the expected format
      return searchResults.map((result) => ({
        id: result.id,
        userId: result.userId,
        agentId: result.agentId,
        content: result.content,
        roomId: result.roomId,
        createdAt: result.createdAt,
        knowledgeFileKey: result.knowledgeFileKey,
        similarityScore: parseFloat(result.similarity),
      }));
    } catch (error) {
      elizaLogger.error(
        `Error searching knowledge by embedding: ${error.message}`
      );
      return [];
    }
  }

  /**
   * Query knowledge across multiple rooms with relevance to a given text.
   * @param queryText The text to search for relevant knowledge.
   * @param roomIds Optional array of room IDs to search in. If not provided, searches across all rooms.
   * @param limit The maximum number of results to return.
   * @param threshold The minimum similarity threshold.
   * @returns An array of knowledge items formatted for use in the runtime.
   */
  async queryKnowledge(
    queryText: Memory,
    roomIds?: string[],
    limit = 10,
    threshold = 0.7
  ): Promise<KnowledgeItem[]> {
    try {
      // If no room IDs provided, query across all rooms the agent participates in
      if (!roomIds || roomIds.length === 0) {
        roomIds = await this.runtime.databaseAdapter.getRoomsForParticipant(
          this.runtime.agentId
        );
      }

      if (!roomIds || roomIds.length === 0) {
        elizaLogger.info("No rooms available for knowledge query.");
        return [];
      }

      // Generate embedding for the query text
      const queryEmbedding =
        queryText.embedding ||
        (await (
          await this.runtime.messageManager.addEmbeddingToMemory(queryText)
        ).embedding);
      if (!queryEmbedding || queryEmbedding.length === 0) {
        elizaLogger.error("Failed to generate embedding for query text.");
        return [];
      }

      const queryEmbeddingArray = `ARRAY[${queryEmbedding.join(",")}]`;
      const roomIdsPlaceholders = roomIds.map(() => "?").join(",");

      // Query across all specified rooms
      const searchResults: AnyType[] = await new Promise((resolve, reject) => {
        this.db.all(
          `SELECT id, userId, agentId, content, roomId, createdAt,
              1 - (embedding <-> ${queryEmbeddingArray}) AS similarity 
           FROM knowledge 
           WHERE roomId IN (${roomIdsPlaceholders}) AND similarity > ?
           ORDER BY similarity DESC 
           LIMIT ?;`,
          // @ts-expect-error test
          [...roomIds, threshold, limit],
          (err, res) => {
            if (err) reject(err);
            else resolve(res || []);
          }
        );
      });

      if (!searchResults || searchResults.length === 0) {
        elizaLogger.info("No matching knowledge found across specified rooms.");
        return [];
      }

      // Transform results into KnowledgeItem format
      return searchResults.map((result) => {
        // Parse the content string back into a Content object
        let contentObj: AnyType;
        try {
          contentObj =
            typeof result.content === "string"
              ? JSON.parse(result.content)
              : result.content;

          // Ensure it has at least a text property
          if (!contentObj.text && typeof contentObj === "string") {
            contentObj = { text: contentObj };
          }
        } catch (e) {
          // If parsing fails, treat the content as plain text
          contentObj = { text: result.content };
        }

        return {
          id: result.id,
          content: contentObj,
          similarity: parseFloat(result.similarity),
        };
      });
    } catch (error) {
      elizaLogger.error(`Error querying knowledge: ${error.message}`);
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
   * Starts the periodic knowledge syncing.
   * @param intervalMs The interval in milliseconds for syncing knowledge.
   * @param batchSizeKB The maximum size of each batch in kilobytes.
   */
  public startPeriodicSync(intervalMs = 2 * 60 * 1000, batchSizeKB = 4): void {
    if (this.syncInterval) {
      elizaLogger.warn("Knowledge sync is already running.");
      return;
    }

    elizaLogger.info(
      `Starting periodic knowledge sync (every ${intervalMs / 1000}s)...`
    );
    this.syncInterval = setInterval(async () => {
      try {
        // First retrieve and process any new knowledge files
        await this.retrieveAndProcessKnowledgeFiles(this.alias);

        // Then sync any new memories to Recall
        await this.syncKnowledgeToRecall(this.alias, batchSizeKB);
      } catch (error) {
        elizaLogger.error(`Periodic knowledge sync failed: ${error.message}`);
      }
    }, intervalMs);

    // Perform an immediate sync on startup
    Promise.all([
      this.retrieveAndProcessKnowledgeFiles(this.alias),
      this.syncKnowledgeToRecall(this.alias, batchSizeKB),
    ]).catch((error) =>
      elizaLogger.error(`Initial knowledge sync failed: ${error.message}`)
    );
  }

  /**
   * Stops the periodic knowledge syncing.
   */
  public stopPeriodicSync(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = undefined;
      elizaLogger.info("Stopped periodic knowledge syncing.");
    }
  }

  // 2. Fix the processing of existing Parquet files
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
      elizaLogger.info(`Processing knowledge file: ${knowledgeFile}`);
      const fileData = await this.client
        .bucketManager()
        .get(bucketAddress, knowledgeFile);

      if (!fileData.result) {
        elizaLogger.warn(
          `Invalid or empty result for knowledge file: ${knowledgeFile}`
        );
        return;
      }

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
        `Parquet buffer created, size: ${parquetBuffer.length} bytes`
      );
      const reader = await ParquetReader.openBuffer(parquetBuffer);
      const cursor = reader.getCursor();

      let recordCount = 0;
      let record: AnyType;

      while ((record = await cursor.next())) {
        recordCount++;

        // Debug record structure to understand what fields are available
        if (recordCount === 1) {
          elizaLogger.debug(
            `First record structure: ${Object.keys(record).join(", ")}`
          );
        }

        // Extract data from Chain of Thought record - more permissive checking
        const userId = record.userId;
        const agentId = record.agentId;
        const embedding = record.embedding;

        // Skip records that are definitely missing critical fields
        if (!userId || !agentId || !embedding || !Array.isArray(embedding)) {
          elizaLogger.warn(
            `Record missing critical fields in ${knowledgeFile}, skipping`,
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
          // Get text content from either userMessage or log field
          const messageText = record.userMessage || "";
          const logText = record.log || "";
          const text = messageText || logText || "No content available";

          // Create Content object
          const content: Content = { text };

          // Generate a deterministic ID based on record data
          const idBase = `${userId}-${agentId}-${record.timestamp || Date.now()}`;
          const memoryId = stringToUuid(idBase);

          // Create a room ID from userId and agentId
          const roomId = stringToUuid(`room-${userId}-${agentId}`);

          // Convert timestamp string to number if available
          let timestamp: number;
          try {
            // Try to parse the timestamp if it exists
            timestamp = record.timestamp
              ? new Date(record.timestamp).getTime()
              : Date.now();
          } catch (e) {
            // Fall back to current time if parsing fails
            timestamp = Date.now();
            elizaLogger.warn(
              `Failed to parse timestamp in record: ${e.message}`
            );
          }

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
            `Error processing record in ${knowledgeFile}: ${recordError.message}`,
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
        `Processed ${recordCount} records from file ${knowledgeFile}`
      );
      await this.markFileAsProcessed(knowledgeFile);
    } catch (error) {
      elizaLogger.error(
        `Error processing file ${knowledgeFile}: ${error.message}`,
        { error, stack: error.stack }
      );
      throw error;
    }
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

      // If we have a specific room context, use it
      const roomId = message.roomId;
      const roomIds = roomId ? [roomId] : undefined;

      // Query for relevant knowledge
      const relevantKnowledge = await this.queryKnowledge(message, roomIds);

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
