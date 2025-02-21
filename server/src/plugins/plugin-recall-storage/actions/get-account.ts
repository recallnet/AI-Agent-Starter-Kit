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

const accountInfoKeywords = [
  "account data",
  "account info",
  "check my account",
  "get my account",
  "account details",
  "retrieve account",
  "what's my account",
];

export const getAccountInfoAction: Action = {
  name: "GET_ACCOUNT_INFO",
  similes: [
    "GET_ACCOUNT_INFO",
    "CHECK_ACCOUNT",
    "ACCOUNT_INFORMATION",
    "ACCOUNT_DETAILS",
    "RETRIEVE_ACCOUNT_INFO",
  ],
  validate: async (_runtime: IAgentRuntime, message: Memory) => {
    const text = message.content.text.toLowerCase();

    // Ensure the user is specifically requesting account info
    if (!accountInfoKeywords.some((keyword) => text.includes(keyword))) {
      return false;
    }

    elizaLogger.info("GET_ACCOUNT_INFO Validation Passed!");
    return true;
  },
  description: "Retrieves the user's Recall account information.",
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
      elizaLogger.info("Fetching account information...");
      const accountInfo = await recallService.getAccountInfo();

      if (accountInfo) {
        const { address, nonce, balance, parentBalance } = accountInfo;
        const formattedBalance =
          balance !== undefined ? `${balance.toString()} credits` : "Unknown";
        const formattedParentBalance =
          parentBalance !== undefined
            ? `${parentBalance.toString()} credits`
            : "N/A";

        elizaLogger.info(
          `Account Info Retrieved: Address: ${address}, Balance: ${formattedBalance}`
        );

        text = `📜 **Your Recall Account Information:**\n\n🔹 **Address:** ${address}\n🔹 **Nonce:** ${nonce}\n🔹 **Balance:** ${formattedBalance}\n🔹 **Parent Balance:** ${formattedParentBalance}`;
      } else {
        elizaLogger.error("GET_ACCOUNT_INFO failed: No account info received.");
        text =
          "⚠️ Unable to retrieve your account information. Please try again later.";
      }
    } catch (error) {
      elizaLogger.error(`GET_ACCOUNT_INFO error: ${error.message}`);
      text =
        "⚠️ An error occurred while fetching your account information. Please try again later.";
    }

    // Create a new memory entry for the response
    const newMemory: Memory = {
      ...message,
      userId: message.agentId,
      content: {
        text,
        action: "GET_ACCOUNT_INFO",
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
        content: { text: "Check my account data" },
      },
      {
        user: "{{agentName}}",
        content: {
          text: "📜 **Your Recall Account Information:**\n🔹 **Address:** 0x123...456\n🔹 **Nonce:** 5\n🔹 **Balance:** 100 credits\n🔹 **Parent Balance:** 500 credits",
          action: "GET_ACCOUNT_INFO",
        },
      },
    ],
    [
      {
        user: "{{user1}}",
        content: { text: "Get my account information" },
      },
      {
        user: "{{agentName}}",
        content: {
          text: "📜 **Your Recall Account Information:**\n🔹 **Address:** 0xABC...DEF\n🔹 **Nonce:** 3\n🔹 **Balance:** 200 credits\n🔹 **Parent Balance:** N/A",
          action: "GET_ACCOUNT_INFO",
        },
      },
    ],
    [
      {
        user: "{{user1}}",
        content: { text: "Retrieve my account details" },
      },
      {
        user: "{{agentName}}",
        content: {
          text: "📜 **Your Recall Account Information:**\n🔹 **Address:** 0x789...XYZ\n🔹 **Nonce:** 2\n🔹 **Balance:** 50 credits\n🔹 **Parent Balance:** 300 credits",
          action: "GET_ACCOUNT_INFO",
        },
      },
    ],
  ] as ActionExample[][],
};
