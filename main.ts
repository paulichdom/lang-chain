import 'dotenv/config';
import { RunnableSequence } from '@langchain/core/runnables';

import {
  loadAndSplitChunks,
  initializeVectorstoreWithDocuments,
} from './lib/helpers.ts';

const splitDocs = await loadAndSplitChunks({
  chunkSize: 1536,
  chunkOverlap: 128,
});

const vectorstore = await initializeVectorstoreWithDocuments({
  documents: splitDocs,
});

const retriever = vectorstore.asRetriever();

const convertDocsToString = (documents: Document[]): string => {
  return documents
    .map((document) => `<doc>\n${document.pageContent}\n</doc>`)
    .join('\n');
};

const documentRetrievalChain = RunnableSequence.from([
  (input) => input.standalone_question,
  retriever,
  convertDocsToString,
]);

import { createRephraseQuestionChain } from './lib/helpers.ts';

const rephraseQuestionChain = createRephraseQuestionChain();

import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';

const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher,
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of your ability
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ['system', ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder('history'),
  [
    'human',
    `Now, answer this question using the previous context and chat history:
  
    {standalone_question}`,
  ],
]);

import { RunnablePassthrough } from '@langchain/core/runnables';
import { ChatOpenAI } from '@langchain/openai';

const conversationalRetrievalChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    standalone_question: rephraseQuestionChain,
  }),
  RunnablePassthrough.assign({
    context: documentRetrievalChain,
  }),
  answerGenerationChainPrompt,
  new ChatOpenAI({ modelName: 'gpt-3.5-turbo-1106' }),
]);

import { HttpResponseOutputParser } from 'langchain/output_parsers';

// "text/event-stream" is also supported
const httpResponseOutputParser = new HttpResponseOutputParser({
  contentType: 'text/plain',
});

import { RunnableWithMessageHistory } from '@langchain/core/runnables';
import { ChatMessageHistory } from 'langchain/stores/message/in_memory';

const messageHistory = new ChatMessageHistory();

/* const finalRetrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: (_sessionId) => messageHistory,
  historyMessagesKey: "history",
  inputMessagesKey: "question",
}).pipe(httpResponseOutputParser); */

const messageHistories = {};

const getMessageHistoryForSession = (sessionId) => {
  if (messageHistories[sessionId] !== undefined) {
    return messageHistories[sessionId];
  }
  const newChatSessionHistory = new ChatMessageHistory();
  messageHistories[sessionId] = newChatSessionHistory;
  return newChatSessionHistory;
};

const finalRetrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: getMessageHistoryForSession,
  inputMessagesKey: 'question',
  historyMessagesKey: 'history',
}).pipe(httpResponseOutputParser);

const port = 8087;

const handler = async (request: Request): Promise<Response> => {
  const body = await request.json();
  const stream = await finalRetrievalChain.stream(
    {
      question: body.question,
    },
    { configurable: { sessionId: body.session_id } }
  );

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/plain',
    },
  });
};

Deno.serve({ port }, handler);

const decoder = new TextDecoder();

// readChunks() reads from the provided reader and yields the results into an async iterable
function readChunks(reader) {
  return {
    async *[Symbol.asyncIterator]() {
      let readResult = await reader.read();
      while (!readResult.done) {
        yield decoder.decode(readResult.value);
        readResult = await reader.read();
      }
    },
  };
}

// First question

/* const response = await fetch(`http://localhost:${port}`, {
    method: "POST",
    headers: {
        "content-type": "application/json",
    },
    body: JSON.stringify({
        question: "What are the prerequisites for this course?",
        session_id: "1", // Should randomly generate/assign
    })
});

// response.body is a ReadableStream
const reader = response.body?.getReader();

for await (const chunk of readChunks(reader)) {
  console.log("CHUNK:", chunk);
} */

// Second question

/* const response1 = await fetch(`http://localhost:${port}`, {
  method: "POST",
  headers: {
    "content-type": "application/json",
  },
  body: JSON.stringify({
    question: "Can you list them in bullet point format?",
    session_id: "1", // Should randomly generate/assign
  })
});

// response.body is a ReadableStream
const reader1 = response1.body?.getReader();

for await (const chunk of readChunks(reader1)) {
  console.log("CHUNK:", chunk);
} */

// Third question

const response = await fetch(`http://localhost:${port}`, {
  method: "POST",
  headers: {
    "content-type": "application/json",
  },
  body: JSON.stringify({
    question: "What did I just ask you?",
    session_id: "2", // Should randomly generate/assign
  })
});

// response.body is a ReadableStream
const reader = response.body?.getReader();

for await (const chunk of readChunks(reader)) {
  console.log("CHUNK:", chunk);
}