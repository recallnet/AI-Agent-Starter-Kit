import { Plugin } from "@ai16z/eliza";
import { recallCotProvider } from "./providers/index.js";
// import { RecallService } from "./services/recall.service.js";
import { buyCreditAction } from "./actions/buy-credit.js";
import { getCreditBalanceAction } from "./actions/get-balance.js";
import { getAccountInfoAction } from "./actions/get-account.js";
import { listBucketsAction } from "./actions/list-buckets.js";
import { createBucketAction } from "./actions/create-bucket.js";
import { addObjectAction } from "./actions/add-object.js";
import { getObjectAction } from "./actions/get-object.js";
import { gateDataAction } from "./actions/gate-action.js";
import { knowledgeEvaluator } from "./evaluators/knowledge.js";

export const recallStoragePlugin: Plugin = {
  name: "Recall Storage Plugin",
  description: "Provides basic Recall storage functionality",
  actions: [
    buyCreditAction,
    getCreditBalanceAction,
    getAccountInfoAction,
    listBucketsAction,
    gateDataAction,
    addObjectAction,
    getObjectAction,
    createBucketAction,
  ],
  providers: [recallCotProvider],
  services: [],
  evaluators: [knowledgeEvaluator],
};

export default recallStoragePlugin;
