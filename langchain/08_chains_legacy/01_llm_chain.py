"""
LLMCHAIN
==========
LLMChain is the original, most basic legacy chain: it binds a PromptTemplate
to an LLM so that calling `.run()` or `.invoke()` formats the prompt with
your inputs, sends it to the model, and returns the parsed output. It predates
LCEL (`prompt | llm`) and is now considered legacy, but is still extremely
common in older codebases and tutorials.

Modern LangChain recommends `prompt | llm | output_parser` instead, but
understanding LLMChain helps when reading/maintaining existing projects.
"""

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["product"],
    template="Suggest a catchy, one-line slogan for a company that makes {product}.",
)

chain = LLMChain(llm=llm, prompt=prompt)

# .run() is the classic single-input/single-output shorthand
slogan = chain.run(product="eco-friendly water bottles")
print(slogan)
# -> "Hydrate Responsibly, Live Sustainably."

# .invoke() is the Runnable-compatible entry point (LLMChain implements Runnable)
result = chain.invoke({"product": "noise-cancelling headphones"})
print(result)
# -> {"product": "noise-cancelling headphones", "text": "Silence the World, Amplify Your Focus."}

# LLMChain also supports batch, mirroring the Runnable interface
batch_results = chain.batch(
    [{"product": "electric bikes"}, {"product": "instant coffee"}]
)
for r in batch_results:
    print(r["text"])
# -> "Ride the Future, Silently."
# -> "Great Coffee, Zero Wait."

# verbose=True prints the exact formatted prompt and raw LLM output — handy for debugging
debug_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
debug_chain.run(product="standing desks")
# -> prints the full rendered prompt to stdout before returning the slogan
