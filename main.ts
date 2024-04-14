import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo-1106',
});

const response = await model.invoke([new HumanMessage('Wirite a haiku')]);

console.log({ response });
