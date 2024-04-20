// Loading

import { GithubRepoLoader } from 'langchain/document_loaders/web/github';
// Peer dependency, used to support .gitignore syntax
import ignore from 'ignore';

// Will not include anything under "ignorePaths"
const loader = new GithubRepoLoader(
  'https://github.com/langchain-ai/langchainjs',
  { recursive: false, ignorePaths: ['*.md', 'yarn.lock'] }
);

const docs = await loader.load();

// Peer dependency
import * as parse from 'pdf-parse';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';

const loader1 = new PDFLoader('./data/MachineLearning-Lecture01.pdf');

const rawCS229Docs = await loader1.load();

// Splitting

import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const splitter = RecursiveCharacterTextSplitter.fromLanguage('js', {
  chunkSize: 32,
  chunkOverlap: 0,
});

const code = `function helloWorld() {
  console.log("Hello, World!");
  }
  // Call the function
  helloWorld();`;

const result = await splitter.splitText(code);

import { CharacterTextSplitter } from 'langchain/text_splitter';

const splitter1 = new CharacterTextSplitter({
  chunkSize: 32,
  chunkOverlap: 0,
  separator: ' ',
});

const result1 = await splitter1.splitText(code);

const splitter2 = RecursiveCharacterTextSplitter.fromLanguage('js', {
  chunkSize: 64,
  chunkOverlap: 32,
});

const result2 = await splitter2.splitText(code);

const splitter3 = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 64,
});

const splitDocs = await splitter3.splitDocuments(rawCS229Docs);