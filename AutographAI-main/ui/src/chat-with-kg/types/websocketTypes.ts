export type WebSocketRequest = {
  type: "question";
  question: string;
  api_key?: string;
  model_name?: string;
};

export interface ContextHit {
  name:     string;
  labels:   string[];
  preview:  string;
  score:    number;
}

export interface SemanticResponse {
  type:    "semantic";
  matches: ContextHit[];
}

export type WebSocketResponse =
  | { type: "start" }
  | { type: "stream"; output: string }
  | { type: "end"; output: string; generated_cypher: string | null }
  | { type: "error"; detail: string }
  | SemanticResponse
  | { type: "debug"; detail: string };

export type ConversationState = "waiting" | "streaming" | "ready" | "error" | "semantic";
