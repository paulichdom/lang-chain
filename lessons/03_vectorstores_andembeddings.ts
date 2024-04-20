// Vectorstore ingestion

import { OpenAIEmbeddings } from '@langchain/openai';

const embeddings = new OpenAIEmbeddings();

const result3 = await embeddings.embedQuery('This is some sample text');

import { similarity } from 'ml-distance';

const vector1 = await embeddings.embedQuery(
  'What are vectors useful for in machine learning?'
);
const unrelatedVector = await embeddings.embedQuery(
  'A group of parrots is called a pandemonium.'
);

const result4 = similarity.cosine(vector1, unrelatedVector);

const similarVector = await embeddings.embedQuery(
  'Vectors are representations of information.'
);

const result5 = similarity.cosine(vector1, similarVector);

// Peer dependency
import * as parse from 'pdf-parse';

const splitter4 = new RecursiveCharacterTextSplitter({
  chunkSize: 128,
  chunkOverlap: 0,
});

const splitDocs1 = await splitter4.splitDocuments(rawCS229Docs);

import { MemoryVectorStore } from 'langchain/vectorstores/memory';

const vectorstore = new MemoryVectorStore(embeddings);

const result6 = await vectorstore.addDocuments(splitDocs);

const retrievedDocs = await vectorstore.similaritySearch(
  'What is deep learning?',
  4
);

const pageContents = retrievedDocs.map((doc) => doc.pageContent);

// Retrievers

const retriever = vectorstore.asRetriever();

const result7 = await retriever.invoke('What is deep learning?');
