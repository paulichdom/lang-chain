import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo-1106',
});

const response1 = await model.invoke([new HumanMessage('Wirite a haiku')]);

import { ChatPromptTemplate } from '@langchain/core/prompts';

const prompt = ChatPromptTemplate.fromTemplate(
  `What are three good names for a company that makes {product}?`
);

const response2 = await prompt.format({
  product: 'colorful socks',
});

const response3 = await prompt.formatMessages({
  product: 'colorful socks',
});

import {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
} from '@langchain/core/prompts';

const promptFromMessages = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    'You are an expert at picking company names.'
  ),
  HumanMessagePromptTemplate.fromTemplate(
    'What are three good names for a company that makes {product}?'
  ),
]);

const response4 = await promptFromMessages.formatMessages({
  product: 'shiny objects',
});

const promptFromMessages1 = ChatPromptTemplate.fromMessages([
  ['system', 'You are an expert at picking company names.'],
  ['human', 'What are three good names for a company that makes {product}?'],
]);

const response5 = await promptFromMessages1.formatMessages({
  product: 'shiny objects',
});

const chain = prompt.pipe(model);

const response6 = await chain.invoke({
  product: 'colorful socks',
});

import { StringOutputParser } from '@langchain/core/output_parsers';

const outputParser = new StringOutputParser();

const nameGenerationChain = prompt.pipe(model).pipe(outputParser);

const response7 = await nameGenerationChain.invoke({
  product: 'fancy cookies',
});

import { RunnableSequence } from '@langchain/core/runnables';

const nameGenerationChain1 = RunnableSequence.from([
  prompt,
  model,
  outputParser,
]);

const response8 = await nameGenerationChain1.invoke({
  product: 'fancy cookies',
});

const stream = await nameGenerationChain.stream({
  product: 'really cool robots',
});

for await (const chunk of stream) {
  // console.log(chunk);
}

const inputs = [
  { product: 'large calculators' },
  { product: 'alpaca wool sweaters' },
];

const response9 = await nameGenerationChain.batch(inputs);

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

console.log(splitDocs.slice(0, 5));
