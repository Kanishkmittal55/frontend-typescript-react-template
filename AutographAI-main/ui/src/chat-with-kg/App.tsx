import { useCallback, useEffect, useState, ChangeEvent } from "react";
import Split from "react-split";
import ChatContainer from "./ChatContainer";
import type { ChatMessageObject } from "./ChatMessage";
import ChatInput from "./ChatInput";
import useWebSocket, { ReadyState } from "react-use-websocket";
import KeyModal from "../components/keymodal";
import type {
  ContextHit,
  ConversationState,
  WebSocketRequest,
  WebSocketResponse,
} from "./types/websocketTypes";

const SEND_REQUESTS = true;

const chatMessageObjects: ChatMessageObject[] = SEND_REQUESTS
  ? []
  : [
      {
        id: Date.now() + Math.random(), // Globally unique ID,
        type: "input",
        sender: "self",
        message:
          "This is the first message which has decently long text and would denote something typed by the user",
        complete: true,
      },
      {
        id: Date.now() + Math.random(), // Globally unique ID,
        type: "text",
        sender: "bot",
        message:
          "And here is another message which would denote a response from the server, which for now will only be text",
        complete: true,
      },
    ];

    const chatOllamaMessageObjects: ChatMessageObject[] = SEND_REQUESTS
    ? []
    : [
        {
          id: Date.now() + Math.random(), // Globally unique ID,
          type: "input",
          sender: "self",
          message:
            "This is the first message which has decently long text and would denote something typed by the user",
          complete: true,
        },
        {
          id: Date.now() + Math.random(), // Globally unique ID,
          type: "text",
          sender: "bot",
          message:
            "And here is another message which would denote a response from the server, which for now will only be text",
          complete: true,
        },
      ];

const URI =
import.meta.env.VITE_KG_CHAT_BACKEND_ENDPOINT
?? "ws://localhost:7860/text2text";

  const OLLAMA_URI =
  import.meta.env.VITE_KG_CHAT_BACKEND_ENDPOINT_OLLAMA ??
  "ws://localhost:7860/ollama/text2text";

const HAS_API_KEY_URI =
  import.meta.env.VITE_HAS_API_KEY_ENDPOINT ??
  "http://localhost:7860/hasapikey";

const QUESTIONS_URI =
  import.meta.env.VITE_KG_CHAT_SAMPLE_QUESTIONS_ENDPOINT ??
  "http://localhost:7860/questionProposalsForCurrentDb";

function loadKeyFromStorage() {
  return localStorage.getItem("api_key");
}

const QUESTION_PREFIX_REGEXP = /^[0-9]{1,2}[\w]*[\.\)\-]*[\w]*/;

function stripQuestionPrefix(question: string): string {
  if (question.match(QUESTION_PREFIX_REGEXP)) {
    return question.replace(QUESTION_PREFIX_REGEXP, "");
  }
  return question;
}

function App() {
  const [serverAvailable, setServerAvailable] = useState(true);

  const [compareMode, setCompareMode] = useState(false);

  const [needsApiKeyLoading, setNeedsApiKeyLoading] = useState(true);

  const [needsApiKey, setNeedsApiKey] = useState(true);

  const [chatMessages, setChatMessages] = useState(chatMessageObjects);

  const [chatOllamaMessages, setChatOllamaMessages] = useState(chatOllamaMessageObjects);

  const [conversationState, setConversationState] = useState<ConversationState>("ready");  // ConversationState = "waiting" | "streaming" | "ready" | "error";

  const [ollamaConversationState, setOllamaConversationState] = useState<ConversationState>("ready");  // ConversationState = "waiting" | "streaming" | "ready" | "error";

  // ──────────────────────────────────────────────
  //  semantic‑search / context‑window state
  // ──────────────────────────────────────────────
  const [contextHits, setContextHits] = useState<ContextHit[]>([]);
  const [nGram,       setNGram]       = useState(3);   // UI slider
  const [topK,        setTopK]        = useState(10);  // UI slider

  // useWebSocket hook, manage websocket connection in a react application.
  // On the LHS we have the returned properties , and on the RHS we have the useWebSocket(URI, options) hook 
  const { sendJsonMessage, lastMessage, readyState } = useWebSocket(URI, {
    shouldReconnect: () => true,
    reconnectInterval: 5000,
    onOpen: () => {
      console.log(`✅ WebSocket open on ${URI}`);
    },
    onError: (event) => {
      console.error(`❌ WebSocket error connecting to ${URI}`, event);
    },
  });

  const { sendJsonMessage: sendOllamaJsonMessage, lastMessage: ollamaLastMessage, readyState: ollamaReadyState } = useWebSocket(OLLAMA_URI, {
    shouldReconnect: () => true, // The options
    reconnectInterval: 5000,
  });

  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [ollamaErrorMessage, setOllamaErrorMessage] = useState<string | null>(null);

  const [modalIsOpen, setModalIsOpen] = useState(false);

  const [apiKey, setApiKey] = useState(loadKeyFromStorage() || "");

  const [sampleQuestions, setSampleQuestions] = useState<string[]>([]);

  const [text2cypherModel, setText2cypherModel] = useState<string>("gpt-3.5-turbo-0613");

  const [ollamaText2cypherModel, setOllamaText2cypherModel] = useState<string>("llama3.1");

  const [sampleQuestionIndex, setSampleQuestionIndex] = useState(0);

  const showContent = serverAvailable && !needsApiKeyLoading;

  function loadSampleQuestions() {
    const body = {
      api_key: apiKey,
    };
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    };
    fetch(QUESTIONS_URI, options).then(
      (response) => {
        response.json().then(
          (result) => {
            if (result.output && result.output.length > 0) {
              setSampleQuestions(result.output.map(stripQuestionPrefix));
            } else {
              setSampleQuestions([]);
            }
          },
          (error) => {
            setSampleQuestions([]);
          }
        );
      },
      (error) => {
        setSampleQuestions([]);
      }
    );
  }

  useEffect(() => {
    fetch(HAS_API_KEY_URI).then(
      (response) => {
        response.json().then(
          (result) => {
            // const needsKey = result.output;
            const needsKey = !result.output;
            setNeedsApiKey(needsKey);
            setNeedsApiKeyLoading(false);
            if (needsKey) {
              const api_key = loadKeyFromStorage();
              if (api_key) {
                setApiKey(api_key);
                loadSampleQuestions();
              } else {
                setModalIsOpen(true);
              }
            } else {
              console.log("Reched here")
              loadSampleQuestions();
            }
          },
          (error) => {
            setNeedsApiKeyLoading(false);
            setServerAvailable(false);
          }
        );
      },
      (error) => {
        setNeedsApiKeyLoading(false);
        setServerAvailable(false);
      }
    );
  }, []);

  const TOPK_CONTEXT = 3;          // always request 3 hits

    /** ask backend for semantic hits via the SAME websocket */
  const fetchContext = (q: string) => {
    sendJsonMessage({
      type    : "semantic",
      question: q,
      top_k   : TOPK_CONTEXT
    });
  };

  // Gpt 4 state control
  useEffect(() => {
    if (!lastMessage || !serverAvailable) {
      return;
    }

    const websocketResponse = JSON.parse(lastMessage.data) as WebSocketResponse;
    console.log(websocketResponse.type)
    if (websocketResponse.type === "semantic") {
          setContextHits(websocketResponse.matches || []);
    } else if (websocketResponse.type === "debug") {
      console.log(websocketResponse.detail);
    } else if (websocketResponse.type === "error") {
      setConversationState("error");
      console.log("The openai chat is in error.")
      setErrorMessage(websocketResponse.detail);
      console.error(websocketResponse.detail);
    } else if (websocketResponse.type === "start") {
      console.log("The open ai chat has started")
      setConversationState("streaming");

      setChatMessages((chatMessages) => [
        ...chatMessages,
        {
          id: chatMessages.length,
          type: "text",
          sender: "bot",
          message: "",
          complete: false,
        },
      ]);
    } else if (websocketResponse.type === "stream") {
      console.log("The open ai chat is streaming")
      setChatMessages((chatMessages) => {
        const lastChatMessage = chatMessages[chatMessages.length - 1];
        const rest = chatMessages.slice(0, -1);

        return [
          ...rest,
          {
            ...lastChatMessage,
            message: lastChatMessage.message + websocketResponse.output,
          },
        ];
      });
    } else if (websocketResponse.type === "end") {
      setChatMessages((chatMessages) => {
        const lastChatMessage = chatMessages[chatMessages.length - 1];
        const rest = chatMessages.slice(0, -1);
        return [
          ...rest,
          {
            ...lastChatMessage,
            complete: true,
            cypher: websocketResponse.generated_cypher,
          },
        ];
      });
      setConversationState("ready");
    }
  }, [lastMessage]);

  // Ollama state control
  useEffect(() => {
    if (!ollamaLastMessage || !serverAvailable) {
      return;
    }

    const websocketResponse = JSON.parse(ollamaLastMessage.data) as WebSocketResponse;
    console.log(websocketResponse)

    if (websocketResponse.type === "debug") {
      console.log(websocketResponse.detail);
    } else if (websocketResponse.type === "error") {
      console.log("The ollama chat is in error.")
      setOllamaConversationState("error");
      setOllamaErrorMessage(websocketResponse.detail);
      console.error(websocketResponse.detail);
    } else if (websocketResponse.type === "start") {
      console.log("The ollama chat has started")
      setOllamaConversationState("streaming");

      setChatOllamaMessages((chatOllamaMessages) => [
        ...chatOllamaMessages,
        {
          id: chatOllamaMessages.length,
          type: "text",
          sender: "bot",
          message: "",
          complete: false,
        },
      ]);
    } else if (websocketResponse.type === "stream") {
      setChatOllamaMessages((chatOllamaMessages) => {
        const lastChatMessage = chatOllamaMessages[chatOllamaMessages.length - 1];
        const rest = chatOllamaMessages.slice(0, -1);

        return [
          ...rest,
          {
            ...lastChatMessage,
            message: lastChatMessage.message + websocketResponse.output,
          },
        ];
      });
    } else if (websocketResponse.type === "end") {
      setChatOllamaMessages((chatOllamaMessages) => {
        
        const lastChatMessage = chatOllamaMessages[chatOllamaMessages.length - 1];
        console.log("The last chat message is",lastChatMessage)
        const rest = chatOllamaMessages.slice(0, -1);
        console.log("The rest is " , rest)
        return [
          ...rest,
          {
            ...lastChatMessage,
            complete: true,
            cypher: websocketResponse.generated_cypher,
          },
        ];
      });
      setOllamaConversationState("ready");
    }
  }, [ollamaLastMessage]);

  // Use effect for gpt-4 
  useEffect(() => {
    if (conversationState === "error") {
      const timeout = setTimeout(() => {
        setConversationState("ready");
      }, 1000);
      return () => clearTimeout(timeout);
    }
  }, [conversationState]);

  // Use Effect for Llama 3.1
  useEffect(() => {
    if (ollamaConversationState === "error") {
      const timeout = setTimeout(() => {
        setOllamaConversationState("ready");
      }, 1000);
      return () => clearTimeout(timeout);
    }
  }, [ollamaConversationState]);

  const sendQuestion = (question: string) => {
    const webSocketRequest: WebSocketRequest = {
      type: "question",
      question: question,
    };
    console.log("The open ai web socket response is : ", webSocketRequest.question)
    if (serverAvailable && !needsApiKeyLoading && needsApiKey && apiKey) {
      webSocketRequest.api_key = apiKey;
    }
    webSocketRequest.model_name = text2cypherModel;
    sendJsonMessage(webSocketRequest);
  };

  const sendOllamaQuestion = (question: string) => {
    const webSocketRequest: WebSocketRequest = {
      type: "question",
      question: question,
    };
    console.log("The web socket response is : ", webSocketRequest.question)
    if (serverAvailable && !needsApiKeyLoading && needsApiKey && apiKey) {
      webSocketRequest.api_key = apiKey;
    }
    webSocketRequest.model_name = ollamaText2cypherModel;
    let further_message = sendOllamaJsonMessage(webSocketRequest)
    console.log(further_message)
    sendOllamaJsonMessage(webSocketRequest);
  };

  const onChatInput = (message: string) => {
    if (conversationState !== "ready") return;
  
    // 1) show the user’s message
    setChatMessages((msgs) => [
      ...msgs,
      {
        id:    msgs.length,
        type:  "input",
        sender:"self",
        message,
        complete: true,
      },
    ]);
  
    // 2) switch to “waiting”
    setConversationState("waiting");
    setErrorMessage(null);
  
    // 3) ask for semantic hits
    sendJsonMessage({
      type:     "semantic",
      question: message,
      top_k:    TOPK_CONTEXT,   // e.g. 3
    });
  };
  


  // const onChatInput = async (message: string) => {
  //   if (conversationState === "ready") {
  //     // Add user message to the chat
  //     setChatMessages((chatMessages) =>
  //       chatMessages.concat([
  //         {
  //           id: Date.now() + Math.random(),
  //           type: "input",
  //           sender: "self",
  //           message: message,
  //           complete: true,
  //         },
  //       ])
  //     );

  //     if (SEND_REQUESTS) {
  //       setConversationState("waiting");
  //       try {
  //         // Call FastAPI endpoint for OpenAI
  //         const openaiResponse = await fetch("http://localhost:7860/text2text", {
  //           method: "POST",
  //           headers: {
  //             "Content-Type": "application/json",
  //           },
  //           body: JSON.stringify({
  //             model: "auto", // Model name
  //             prompt: message, // The user's message
  //           }),
  //         });

  //         fetchContext(message)

  //         // Parse the response
  //         if (openaiResponse.ok) {
  //           const data = await openaiResponse.json();

  //           // Add OpenAI's response to the chat
  //           setChatMessages((chatMessages) => [
  //             ...chatMessages,
  //             {
  //               id: chatMessages.length,
  //               type: "text",
  //               sender: "openai",
  //               message: data,
  //               complete: true,
  //             },
  //           ]);
  //         } else {
  //           console.error("Error with OpenAI API:", await openaiResponse.text());
  //         }
  //       } catch (error) {
  //         console.error("Error with OpenAI:", error);
  //       }
  //       setErrorMessage(null);
  //       setConversationState("ready");
  //     }
  //   }

    
  // };

  const onOllamaChatInput = async (message: string) => {
    if (ollamaConversationState === "ready") {
      // Add user message to the chat
      setChatOllamaMessages((chatOllamaMessages) =>
        chatOllamaMessages.concat([
          {
            id: Date.now() + Math.random(),
            type: "input",
            sender: "self",
            message: message,
            complete: true,
          },
        ])
      );
  
      if (SEND_REQUESTS) {
        setOllamaConversationState("waiting");
        try {
            // Call FastAPI endpoint for Ollama
            const ollamaResponse = await fetch("http://localhost:7860/ollama/cyphered", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                model: "auto", // Model name
                prompt: message, // The user's message
              }),
            });
  
            // Parse the response
            if (ollamaResponse.ok) {
              const data = await ollamaResponse.json();
              // Navigate the structure of the backend's return to extract content
              const ollamaMessage =
              data?.generated_text?.message?.content ?? "No content available";

              console.log("Extracted message content:", ollamaMessage);
  
              // Add Ollama's response to the chat
              setChatOllamaMessages((chatOllamaMessages) => [
                ...chatOllamaMessages,
                {
                  id: chatOllamaMessages.length,
                  type: "text",
                  sender: "ollama",
                  message: ollamaMessage,
                  complete: true,
                },
              ]);
            } else {
              console.error("Error with Ollama API:", await ollamaResponse.text());
            }
          } catch (error) {
            console.error("Error with Ollama:", error);
          }
        setErrorMessage(null);
        setOllamaConversationState("ready");
      }
    }

    // if (ollamaConversationState === "ready") {
    //   setChatOllamaMessages((chatOllamaMessages) =>
        
    //     chatOllamaMessages.concat([
    //       {
    //         id: chatMessages.length,
    //         type: "input",
    //         sender: "self",
    //         message: message,
    //         complete: true,
    //       },
    //     ])
    //   );
    //   if (SEND_REQUESTS) {
    //     console.log("Message sent : ", message)
    //     setOllamaConversationState("waiting");
    //     console.log("The ollama conversation state has been set to waiting ")
    //     sendOllamaQuestion(message);
    //   }
    //   setErrorMessage(null);
    // }
  };
  

  useEffect(() => {
    setSampleQuestionIndex(0);
  }, [sampleQuestions])

  const openModal = () => {
    setModalIsOpen(true);
  };

  const sampleQuestionLeft = () => {
    if (sampleQuestionIndex > 0) {
      setSampleQuestionIndex(sampleQuestionIndex - 1);
    }
  };

  const sampleQuestionRight = () => {
    if (sampleQuestionIndex < sampleQuestions.length - 1) {
      setSampleQuestionIndex(sampleQuestionIndex + 1);
    }
  };

  const onSampleQuestionClick = () => {
    const sampleQuestion = sampleQuestions[sampleQuestionIndex];
    if (onChatInput && sampleQuestion !== undefined) {
      onChatInput(sampleQuestion);
    }
  };

  const onCloseModal = () => {
    setModalIsOpen(false);
    if (apiKey && sampleQuestions.length === 0) {
      loadSampleQuestions();
    }
  };

  const onApiKeyChange = (newApiKey: string) => {
    setApiKey(newApiKey);
    localStorage.setItem("api_key", newApiKey);
  };

  const handleModelChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setText2cypherModel(e.target.value)
  }




  return (

      <Split
        className="flex h-screen overflow-hidden"
        sizes={[40, 60]}          // graph | context | chat
        minSize={[320, 350]}
        gutterSize={8}
      >
    
        {/* ───────── LEFT  –  refined graph viewer ───────── */}
        <div className="flex flex-col bg-gray-900">
          <div className="px-3 py-2 bg-gray-800 text-white text-sm font-semibold">
            Refined Graph
          </div>
          <iframe
            src="http://localhost:7860/static/src/Refined-Graph.html"
            className="flex-1 border-0"
            title="Refined Graph"
          />
        </div>
    
        {/* ───────── RIGHT  –  chat (unchanged) ───────── */}
        <div className="flex flex-col">
          {/* Window Header */}
            <div className="flex justify-between items-center bg-gray-800 text-white p-2">
            <span>Chat Window</span>
            <select
              value={text2cypherModel}
              onChange={handleModelChange}
              className="bg-gray-100 border border-gray-300 text-gray-700 py-1 px-2 rounded focus:outline-none"
            >
              <option value="gpt-4o">gpt-3.5-turbo</option>
              <option value="gpt-4">gpt-4</option>
            </select>
            </div>

            {/* Chat Messages */}
            <div className="flex-grow overflow-y-auto p-4 bg-white border-t border-gray-300">
            {!serverAvailable && <div>Server is unavailable, please reload the page to try again.</div>}
            {serverAvailable && needsApiKeyLoading && <div>Initializing...</div>}
            <KeyModal
              isOpen={showContent && needsApiKey && modalIsOpen}
              onCloseModal={onCloseModal}
              onApiKeyChanged={onApiKeyChange}
              apiKey={apiKey}
            />
            {showContent && readyState === ReadyState.OPEN && (
              // We provide the same component the messages generated from the gpt 4 api
              <ChatContainer chatMessages={chatMessages} loading={conversationState === "waiting"} />
            )}
            {showContent && readyState === ReadyState.CONNECTING && <div>Connecting...</div>}
            {showContent && readyState === ReadyState.CLOSED && (
              <div className="flex flex-col">
                <div>Could not connect to server, reconnecting...</div>
              </div>
            )}
            </div>

            {/* Input Section */}
            <div className="bg-gray-100 border-t border-gray-300 p-4">
            <ChatInput
              onChatInput={onChatInput}
              loading={conversationState === "waiting"}
              sampleQuestions={sampleQuestions}
            />
            {sampleQuestions && sampleQuestions.length > 0 && (
              <div className="flex items-center justify-between mt-2 bg-gray-200 rounded-md px-4 py-2 shadow-md">
                <button
                  className="ndl-icon-btn ndl-large"
                  onClick={sampleQuestionLeft}
                  disabled={sampleQuestionIndex <= 0}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#000"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="15 18 9 12 15 6" />
                  </svg>
                </button>
                <div
                  className="flex-grow text-center truncate text-sm"
                  style={{
                    fontSize: "clamp(12px, 2vw, 16px)", // Adjust font size dynamically
                  }}
                >
                  {sampleQuestions[sampleQuestionIndex]}
                </div>
                <button
                  className="ndl-icon-btn ndl-large"
                  onClick={sampleQuestionRight}
                  disabled={sampleQuestionIndex >= sampleQuestions.length - 1}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#000"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                </button>
              </div>
            )}
            </div>
        </div>
    
      </Split>
  );
    
}

export default App;
