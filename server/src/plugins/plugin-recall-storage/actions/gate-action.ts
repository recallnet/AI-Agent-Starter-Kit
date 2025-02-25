import {
  type Action,
  type IAgentRuntime,
  type Memory,
  type State,
  type HandlerCallback,
  type ActionExample,
  elizaLogger,
  ServiceType,
} from "@ai16z/eliza";
import { RecallService } from "../services/recall.service.js";

const gateDataKeywords = [
  "encrypt my data",
  "protect the data",
  "secure my information",
  "store this data",
  "encrypt this information",
  "protect my data",
  "secure this information",
  "gate this data",
  "encrypt conversation",
  "store securely",
  "protect my data",
];

export const gateDataAction: Action = {
  name: "GATE_DATA",
  similes: [
    "GATE_DATA",
    "ENCRYPT_DATA",
    "PROTECT_DATA",
    "SECURE_INFORMATION",
    "STORE_SECURELY",
  ],
  validate: async (_runtime: IAgentRuntime, message: Memory) => {
    const text = message.content.text.toLowerCase();

    // Ensure the user is requesting data protection
    if (!gateDataKeywords.some((keyword) => text.includes(keyword))) {
      return false;
    }

    elizaLogger.info("GATE_DATA Validation Passed!");
    return true;
  },
  description: "Encrypts and securely stores important data using Recall.",
  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
    state?: State,
    _options?: { [key: string]: unknown },
    callback?: HandlerCallback
  ): Promise<boolean> => {
    const recallService = runtime.services.get(
      "recall" as ServiceType
    ) as RecallService;
    let text = "";

    try {
      let currentState = state;
      if (!currentState) {
        currentState = (await runtime.composeState(message)) as State;
      } else {
        currentState = await runtime.updateRecentMessageState(currentState);
      }

      elizaLogger.info("Encrypting and storing data...");

      const storageKey = await recallService.writeKnowledgeToRecall(message);

      if (storageKey) {
        elizaLogger.info(
          `Data successfully gated and stored in Recall at key ${storageKey}`
        );
        text = `🔒 Your data has been securely encrypted and stored in Recall at key ${storageKey}`;
      } else {
        elizaLogger.error("GATE_DATA failed: Encryption or storage error.");
        text =
          "⚠️ Unable to secure your data at the moment. Please try again later.";
      }
    } catch (error) {
      elizaLogger.error(`GATE_DATA error: ${error.message}`);
      text =
        "⚠️ An error occurred while encrypting and storing your data. Please try again later.";
    }

    // Create a new memory entry for the response
    const newMemory: Memory = {
      ...message,
      userId: message.agentId,
      content: {
        text,
        action: "GATE_DATA",
        source: message.content.source,
      },
    };

    // Save to memory
    await runtime.messageManager.createMemory(newMemory);

    // Call callback AFTER saving memory
    await callback?.({
      text,
    });

    return true;
  },
  examples: [
    [
      {
        user: "{{user1}}",
        content: { text: "Please protect the data from our conversation" },
      },
      {
        user: "{{agentName}}",
        content: {
          text: "🔒 Your data is being securely encrypted and stored...",
          action: "GATE_DATA",
        },
      },
    ],
    [
      {
        user: "{{user1}}",
        content: { text: "Encrypt this sensitive information" },
      },
      {
        user: "{{agentName}}",
        content: {
          text: "🔒 Your data is being securely encrypted and stored...",
          action: "GATE_DATA",
        },
      },
    ],
    [
      {
        user: "{{user1}}",
        content: { text: "I need this data stored securely" },
      },
      {
        user: "{{agentName}}",
        content: {
          text: "🔒 Your data is being securely encrypted and stored...",
          action: "GATE_DATA",
        },
      },
    ],
  ] as ActionExample[][],
};
