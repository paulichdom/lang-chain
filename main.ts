import 'dotenv/config';

import { loadAndSplitChunks } from './lib/helpers.ts';

const splitDocs = await loadAndSplitChunks({
  chunkSize: 1536,
  chunkOverlap: 128,
});

import { initializeVectorstoreWithDocuments } from './lib/helpers.ts';

const vectorstore = await initializeVectorstoreWithDocuments({
  documents: splitDocs,
});

const retriever = vectorstore.asRetriever();

import { RunnableSequence } from '@langchain/core/runnables';
import { Document } from '@langchain/core/documents';

const convertDocsToString = (documents: Document[]): string => {
  return documents
    .map((document) => {
      return `<doc>\n${document.pageContent}\n</doc>`;
    })
    .join('\n');
};

const documentRetrievalChain = RunnableSequence.from([
  (input) => input.question,
  retriever,
  convertDocsToString,
]);

import { ChatPromptTemplate } from '@langchain/core/prompts';

const TEMPLATE_STRING = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(TEMPLATE_STRING);

import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser } from '@langchain/core/output_parsers';

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo-1106',
});

const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

