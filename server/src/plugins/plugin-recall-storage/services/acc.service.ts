import axios, { AxiosInstance } from "axios";
import fs from "fs";
import path, { resolve } from "path";
import { elizaLogger, getEmbeddingZeroVector, Memory } from "@ai16z/eliza";
import { getCollablandApiUrl } from "../../../utils.js";

const __dirname = path.dirname(new URL(import.meta.url).pathname);
const chainId = 8453;

const OPENAI_EMBEDDINGS = Boolean(process.env.USE_OPENAI_EMBEDDING ?? "false");

// Simplified StorageService that only handles encryption/decryption
export class AccessControlService {
  private static instance: AccessControlService;
  private client: AxiosInstance | null;
  private encryptActionHash: string | null;
  private decryptActionHash: string | null;
  private started: boolean;

  private constructor() {
    this.client = null;
    this.encryptActionHash = null;
    this.decryptActionHash = null;
    this.started = false;
  }

  static getInstance(): AccessControlService {
    if (!AccessControlService.instance) {
      AccessControlService.instance = new AccessControlService();
    }
    return AccessControlService.instance;
  }

  // Initialize the StorageService with encryption capabilities
  async start(): Promise<void> {
    if (this.started) {
      return;
    }
    try {
      this.client = axios.create({
        baseURL: getCollablandApiUrl(),
        headers: {
          "X-API-KEY": process.env.COLLABLAND_API_KEY || "",
          "X-TG-BOT-TOKEN": process.env.TELEGRAM_BOT_TOKEN || "",
          "Content-Type": "application/json",
        },
        timeout: 5 * 60 * 1000,
      });

      // Load Lit Action hashes
      const actionHashes = JSON.parse(
        (
          await fs.readFileSync(
            resolve(
              __dirname,
              "..",
              "..",
              "..",
              "..",
              "..",
              "lit-actions",
              "actions",
              `ipfs.json`
            )
          )
        ).toString()
      );
      this.encryptActionHash = actionHashes["encrypt-action"].IpfsHash;
      this.decryptActionHash = actionHashes["decrypt-action"].IpfsHash;
      this.started = true;
      return;
    } catch (error) {
      console.warn("Error starting StorageService:", error);
    }
  }

  // Check if encryption/decryption is properly configured
  isConfigured(): boolean {
    if (!OPENAI_EMBEDDINGS) {
      elizaLogger.info(
        "[storage.service] Not using OPENAI embeddings. Encryption is disabled."
      );
      return false;
    }

    if (!this.encryptActionHash) {
      elizaLogger.warn(
        "[storage.service] Encrypt action hash is not initialized. Encryption is disabled."
      );
      return false;
    }

    if (!this.decryptActionHash) {
      elizaLogger.warn(
        "[storage.service] Decrypt action hash is not initialized. Encryption is disabled."
      );
      return false;
    }

    if (!this.client) {
      elizaLogger.warn(
        "[storage.service] HTTP client is not initialized. Encryption is disabled."
      );
      return false;
    }

    return true;
  }

  /**
   * Encrypts a string message using Lit Protocol
   * @param message The message to encrypt
   * @returns Promise resolving to {ciphertext, dataToEncryptHash} or null if encryption fails
   */
  async encryptMessage(
    message: string
  ): Promise<{ ciphertext: string; dataToEncryptHash: string } | null> {
    if (!this.isConfigured()) {
      elizaLogger.warn(
        "[storage.service] Service not configured, can't encrypt message"
      );
      return null;
    }

    elizaLogger.debug("[storage.service] Encrypting message");
    try {
      const { data } = await this.client!.post(
        `/telegrambot/executeLitActionUsingPKP?chainId=${chainId}`,
        {
          actionIpfs: this.encryptActionHash,
          actionJsParams: {
            toEncrypt: message,
          },
        }
      );

      if (data?.response?.response) {
        const {
          ciphertext,
          dataToEncryptHash,
          message: responseMessage,
        } = JSON.parse(data.response.response);
        elizaLogger.debug(
          `[storage.service] Encryption message=${responseMessage}`
        );
        elizaLogger.info(`Encrypted message: ${ciphertext}`);
        if (ciphertext && dataToEncryptHash) {
          return { ciphertext, dataToEncryptHash };
        } else {
          throw new Error(`Encryption failed: data=${JSON.stringify(data)}`);
        }
      } else {
        elizaLogger.warn(
          "[storage.service] Did not get any response from lit action to encrypt"
        );
        return null;
      }
    } catch (error) {
      elizaLogger.error("[storage.service] Error encrypting message:", error);
      throw error;
    }
  }

  /**
   * Decrypts a message that was encrypted with Lit Protocol
   * @param ciphertext The encrypted text
   * @param dataToEncryptHash The hash of the original data
   * @returns Promise resolving to the decrypted message or null if decryption fails
   */
  async decryptMessage(
    ciphertext: string,
    dataToEncryptHash: string
  ): Promise<string | null> {
    if (!this.isConfigured()) {
      elizaLogger.warn(
        "[storage.service] Service not configured, can't decrypt message"
      );
      return null;
    }

    try {
      const { data } = await this.client!.post(
        `/telegrambot/executeLitActionUsingPKP?chainId=${chainId}`,
        {
          actionIpfs: this.decryptActionHash,
          actionJsParams: {
            ciphertext,
            dataToEncryptHash,
            chain: "base",
          },
        }
      );

      if (data?.response?.response) {
        const res = JSON.parse(data.response.response);
        elizaLogger.debug(`[storage.service] Decrypt message="${res.message}"`);
        return res.decrypted;
      } else {
        elizaLogger.warn(
          "[storage.service] Failed to retrieve decrypted data",
          data?.response
        );
        return null;
      }
    } catch (error) {
      elizaLogger.error("[storage.service] Error decrypting message:", error);
      return null;
    }
  }

  /**
   * Process a memory object for storage, encrypting its content
   * @param memory The memory object to process
   * @returns A memory object with encrypted content or null if encryption fails
   */
  async prepareMemoryForStorage(memory: Memory): Promise<Memory | null> {
    if (!this.isConfigured()) {
      elizaLogger.warn(
        "[storage.service] Service not configured, returning original memory"
      );
      return memory;
    }

    try {
      // Extract the text content from the memory
      const contentText =
        typeof memory.content === "string"
          ? memory.content
          : memory.content.text || JSON.stringify(memory.content);

      // Encrypt the content
      const encrypted = await this.encryptMessage(contentText);
      if (!encrypted) {
        return null;
      }

      // Create a new memory object with encrypted content
      const encryptedMemory: Memory = {
        ...memory,
        content: {
          ...memory.content,
          text: JSON.stringify(encrypted),
          isEncrypted: true,
        },
      };

      return encryptedMemory;
    } catch (error) {
      elizaLogger.error(
        `[storage.service] Error preparing memory for storage: ${error.message}`
      );
      return null;
    }
  }

  /**
   * Process a stored record, decrypting its content if it's encrypted
   * @param record The record object from storage
   * @returns The record with decrypted content
   */
  async processStoredRecord(record: Memory): Promise<Memory> {
    if (!this.isConfigured()) {
      elizaLogger.debug(
        "[storage.service] Service not configured, returning original record"
      );
      return record;
    }

    try {
      // Check if the content is encrypted
      let content = record.content;
      if (typeof content === "string") {
        try {
          content = JSON.parse(content);
        } catch (e) {
          // Not JSON, assume it's plain text
          return record;
        }
      }

      // If it's not encrypted or doesn't have the right format, return as is
      if (!content.isEncrypted) {
        return record;
      }

      // Parse the encrypted data
      let encryptedData;
      try {
        encryptedData = JSON.parse(content.text);
      } catch (e) {
        elizaLogger.error(
          "[storage.service] Failed to parse encrypted data",
          e
        );
        return record;
      }

      // Decrypt the content
      const { ciphertext, dataToEncryptHash } = encryptedData;
      const decryptedText = await this.decryptMessage(
        ciphertext,
        dataToEncryptHash
      );

      if (!decryptedText) {
        elizaLogger.warn("[storage.service] Failed to decrypt record content");
        return record;
      }

      // Create a new record with decrypted content
      const decryptedRecord = {
        ...record,
        content: {
          ...record.content,
          text: decryptedText,
          isEncrypted: false,
        },
      };

      return decryptedRecord;
    } catch (error) {
      elizaLogger.error(
        `[storage.service] Error processing stored record: ${error.message}`
      );
      return record;
    }
  }

  static isMemoryStorable(memory: Memory): boolean {
    if (OPENAI_EMBEDDINGS && memory?.embedding != getEmbeddingZeroVector()) {
      return true;
    }
    return false;
  }
}

// Helper function to mask embedding data when logging
export const maskEmbedding = (
  key: string,
  value: number[]
): number[] | string => {
  if (key == "embedding") {
    if (value == getEmbeddingZeroVector()) {
      return "[masked zero embedding]";
    } else {
      return "[maskedEmbedding]";
    }
  }
  return value;
};
