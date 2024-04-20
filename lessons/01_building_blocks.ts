import 'dotenv/config';

// Language model

import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo-1106',
});

const response1 = await model.invoke([new HumanMessage('Wirite a haiku')]);

// Prompt template

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

// LangChain Expression Language (LCEL)

const chain = prompt.pipe(model);

const response6 = await chain.invoke({
  product: 'colorful socks',
});

// Output parser

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

// Streaming

const stream = await nameGenerationChain.stream({
  product: 'really cool robots',
});

for await (const chunk of stream) {
  // console.log(chunk);
}

// Batch

const inputs = [
  { product: 'large calculators' },
  { product: 'alpaca wool sweaters' },
];

const response9 = await nameGenerationChain.batch(inputs);
