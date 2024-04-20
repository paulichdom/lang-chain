
// Question answering

import { loadAndSplitChunks } from './lib/helpers.ts';

const splitDocs2 = await loadAndSplitChunks({
  chunkSize: 1536,
  chunkOverlap: 128,
});

import { initializeVectorstoreWithDocuments } from './lib/helpers.ts';

const vectorstore1 = await initializeVectorstoreWithDocuments({
  documents: splitDocs2,
});

const retriever1 = vectorstore1.asRetriever();

// Document retrieval in a chain

import { Document } from '@langchain/core/documents';

const convertDocsToString = (documents: Document[]): string => {
  return documents
    .map((document) => {
      return `<doc>\n${document.pageContent}\n</doc>`;
    })
    .join('\n');
};

/*
{
question: "What is deep learning?"
}
*/

const documentRetrievalChain = RunnableSequence.from([
  (input) => input.question,
  retriever,
  convertDocsToString,
]);

const results = await documentRetrievalChain.invoke({
  question: 'What are the prerequisites for this course?',
});

// Synthesizing a response

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

import { RunnableMap } from '@langchain/core/runnables';

const runnableMap = RunnableMap.from({
  context: documentRetrievalChain,
  question: (input) => input.question,
});

const result8 = await runnableMap.invoke({
  question: 'What are the prerequisites for this course?',
});

// Augmented generation

const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

const answer = await retrievalChain.invoke({
  question: 'What are the prerequisites for this course?',
});

const followupAnswer = await retrievalChain.invoke({
  question: 'Can you list them in bullet point form?',
});

const docs1 = await documentRetrievalChain.invoke({
  question: 'Can you list them in bullet point form?',
});


