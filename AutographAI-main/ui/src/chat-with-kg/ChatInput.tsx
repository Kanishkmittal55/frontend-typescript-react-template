import { useState, useEffect, KeyboardEvent } from "react";
import { encoding_for_model, TiktokenModel } from "tiktoken";

export type ChatInputProps = {
  onChatInput?: (chatText: string) => void;
  loading?: boolean;
  sampleQuestions: string[]
};

//Needed since the types for react don't include enterKeyHint
declare module "react" {
  interface TextareaHTMLAttributes<T> extends HTMLAttributes<T> {
    enterKeyHint?:
      | "enter"
      | "done"
      | "go"
      | "next"
      | "previous"
      | "search"
      | "send";
  }
}

// Function to calculate token count for a specific model
const getTokenCount = (text: string, model: string = "gpt-3.5-turbo"): number => {
  try {
    const encoding = encoding_for_model(model as TiktokenModel); // Load encoding for the specified model
    const tokenCount = encoding.encode(text).length; // Get token count
    encoding.free(); // Free the encoding to prevent memory leaks
    return tokenCount;
  } catch (error) {
    console.error("Error in tokenizing text:", error);
    return 0; // Return 0 tokens if an error occurs
  }
};

// Regex Tokenizer Function
const regexTokenizer = (text: string): number => {
  const tokens = text.match(/\w+|[^\w\s]/g); // Matches words and punctuation
  return tokens ? tokens.length : 0;
};

function ChatInput(props: ChatInputProps) {
  const { onChatInput, loading, sampleQuestions } = props;
  const [inputText, setInputText] = useState("");
  const [tokenCount, setTokenCount] = useState(0);

  

  const onInputKeyPress = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!loading && event.key === "Enter") {
      handleSend();
    }
  };

  const handleSend = () => {
    if (!loading && inputText !== "" && onChatInput) {
      onChatInput(inputText);
      setInputText("");
      setTokenCount(0); // Reset token count on send
    }
  };

  const handleInputChange = (value: string) => {
    setInputText(value);
    setTokenCount(getTokenCount(value)); // Update token count dynamically
  };


  return (
    <div className="flex flex-col max-w-4xl gap-2">
      {/* Token Count Display */}
      <div className="rounded bg-gray-200 p-2 mb-1 text-sm text-gray-700 border border-gray-300 w-fit">
        Token Count: {tokenCount}
      </div>

      <div className="flex flex-row w-full">
        {/* @ts-ignore */}
        <textarea
          enterKeyHint="send"
          onChange={(e) => handleInputChange(e.target.value)}
          onKeyDown={onInputKeyPress}
          disabled={loading}
          value={inputText}
          rows={1}
          className="w-full max-w-full p-3 m-0 overflow-x-hidden overflow-y-auto bg-transparent border rounded-md outline-none resize-none scroll-p-3 focus:ring-0 focus-visible:ring-0 border-palette-neutral-bg-strong"
          placeholder="Ask something about your database"
        ></textarea>
        <button
          className="flex self-center ndl-icon-btn ndl-large"
          onClick={handleSend}
        >
          <div className="ndl-icon">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
              stroke="currentColor"
              aria-hidden="true"
              className="w-6 h-6 text-light-neutral-text-weak"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
              ></path>
            </svg>
          </div>
        </button>
      </div>
      
    </div>
  );
}

export default ChatInput;
