import {
  ActionExample,
  Evaluator,
  IAgentRuntime,
  Memory,
  State,
  elizaLogger,
  ModelClass,
  generateText,
  ServiceType,
  HandlerCallback,
} from "@ai16z/eliza";
import { RecallService } from "../services/recall.service.js";
import { AccessControlService } from "../services/acc.service.js";

export const knowledgeEvaluator: Evaluator = {
  description: "Knowledge evaluator for checking important content in memory",
  similes: ["knowledge checker", "memory evaluator"],
  examples: [
    {
      context: `Actors in the scene:
    {{user1}}: Programmer and decentralized compute specialist.
    {{agentName}}: Agent user interacting with the user.

    Interesting facts about the actors:
    None`,
      messages: [
        {
          user: "{{user1}}",
          content: {
            text: "I'd like to use a Lit Action to allow AI agents to use their PKPs to encrypt and decrypt data without revealing private keys to users.",
          },
        },
        {
          user: "{{user1}}",
          content: {
            text: "The mantis shrimp's eyes have 16 types of photoreceptor cells, allowing them to see ultraviolet and polarized light, far beyond human capabilities.",
          },
        },
        {
          user: "{{user1}}",
          content: {
            text: "Neutron stars are so dense that a sugar-cube-sized piece of one would weigh about a billion tons on Earth.",
          },
        },
      ] as ActionExample[],
      outcome: "TRUE",
    },
  ],
  handler: async (
    runtime: IAgentRuntime,
    memory: Memory,
    state?: State,
    _options?: { [key: string]: unknown },
    callback?: HandlerCallback
  ) => {
    const context = `
    ${JSON.stringify(knowledgeEvaluator.examples[0].messages)}
    \n
    ## Instructions for the agent:
    Determine if the memory contains important content from the participant's query that reveals subject-matter expertise. If the memory is simply a question or a statement that does not reveal subject-matter expertise, the memory is not important.

    ## Examples of not important content:
    - "What can you tell me about cross-chain account management?"
    - "I am interested in learning more about the history of EVM chains."
    - "What are the best available tools for managing secure wallet authentication?"

    ## Examples of important content:
    - "I know that you can use a Lit Action to allow AI agents to use their PKPs to encrypt and decrypt data without revealing private keys to users. This is a great way to ensure that user data is secure and private. How can I implement this feature in my application?"
    - "Did you know that the mantis shrimp's eyes have 16 types of photoreceptor cells, allowing them to see ultraviolet and polarized light, far beyond human capabilities? This is an interesting fact that I recently learned and I thought you might find it interesting as well."
    - "Neutron stars are so dense that a sugar-cube-sized piece of one would weigh about a billion tons on Earth. This is an incredible fact that I recently discovered and I wanted to share it with you."
    - "Cross-chain account management allows users to control and manage their accounts and assets across different blockchain networks"
    - "Cross-chain bridges and protocols are often targeted by attackers, with exploits leading to significant losses in the past."

    Keep in mind that the important content should reveal subject-matter expertise or knowledge that can be of various topics and not just limited to the examples provided above.
    
    Answer only with the following responses:
    - TRUE
    - FALSE

    The following is the memory content you need to evaluate: ${memory.content.text}`;
    const recallService = runtime.services.get(
      "recall" as ServiceType
    ) as RecallService;
    let text = "";
    try {
      let currentState = state;
      if (!currentState) {
        currentState = (await runtime.composeState(memory)) as State;
      } else {
        currentState = await runtime.updateRecentMessageState(currentState);
      }

      // prompt the agent to determine if the memory contains important content
      const res = await generateText({
        runtime,
        context,
        modelClass: ModelClass.SMALL,
      });
      elizaLogger.debug("[knowledge evaluator] Response from the agent:", res);

      const important = res === "TRUE" ? true : false;
      if (important) {
        elizaLogger.log(
          `[knowledge evaluator] Important content found in memory. Storing message with embedding`
        );
        elizaLogger.info("Encrypting and storing data...");

        const storageKey = await recallService.writeKnowledgeToRecall(memory);

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
        // Create a new memory entry for the response
        const newMemory: Memory = {
          ...memory,
          userId: memory.agentId,
          content: {
            text,
            action: "GATE_DATA",
            source: memory.content.source,
          },
        };

        // Save to memory
        await runtime.messageManager.createMemory(newMemory);

        // Call callback AFTER saving memory
        await callback?.({
          text,
        });

        return true;
      }
    } catch (error) {
      elizaLogger.error(`GATE_DATA error: ${error.message}`);
      text =
        "⚠️ An error occurred while encrypting and storing your data. Please try again later.";
    }
    return false;
  },
  name: "knowledgeEvaluator",
  validate: async (_runtime: IAgentRuntime, memory: Memory, state?: State) => {
    // only available if we're able to use remote storage and memory has proper embeddings
    // confirm first that the gate-action plugin has not already stored this memory
    if (
      AccessControlService.getInstance().isConfigured() &&
      !state?.hasGatedAndStored
    ) {
      return AccessControlService.isMemoryStorable(memory);
    }
    return false;
  },
};
