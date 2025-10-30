from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.1,
    top_p=0.9,
)

model = ChatHuggingFace(llm=llm)

# Define structured schema
class Review(TypedDict):
    key_themes = Annotated[list[str], "Provide all key themes in review inside a list"]
    summary: Annotated[str, "A brief summary of review"]
    sentiment: Annotated[Literal["pos","neg"], "Provide sentiment of review"]
    pros: Annotated[Optional[list[str]], "Write all pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write all cons inside a list"]
    name: Annotated[Optional[str], "Write name of reviewer if explicitly mentioned; otherwise None"]

# Wrap model with structured output
structured_model = model.with_structured_output(Review)

# Directly provide review text as input string
result = structured_model.invoke(
  """I recently bought the Apple AirPods Pro 2, and after using them for a few weeks, I can confidently say they’re among the best wireless earbuds I’ve ever owned.
The noise cancellation is phenomenal — it blocks out background chatter, traffic noise, and even the hum of airplane engines. The Transparency Mode feels incredibly natural, allowing me to hear my surroundings without needing to remove the earbuds. Apple has clearly refined this feature since the first generation.
The sound quality is equally impressive. The bass is deep but not overpowering, vocals are clear, and the overall audio balance feels premium. Whether I’m listening to podcasts, music, or watching movies, everything sounds crisp and immersive.
Comfort-wise, they fit perfectly in my ears. Even after long listening sessions, I don’t feel any discomfort or ear fatigue. The new smaller ear tip option makes a noticeable difference for people with smaller ears.
The battery life has improved too — I usually get around 5–6 hours on a single charge, and the MagSafe charging case provides several additional charges. The case’s built-in speaker and Find My integration make it easier to locate if I misplace it.
However, there are a few drawbacks. The price is definitely on the higher side, and while the build quality justifies it to some extent, it still feels a bit steep compared to competitors offering similar features at a lower cost. Also, the AirPods work best within the Apple ecosystem — Android users won’t get access to all the features like automatic switching or spatial audio customization.

✅ Pros
Outstanding noise cancellation and transparency mode
Excellent sound quality with balanced bass and clear vocals
Comfortable fit with multiple ear tip options
Great integration with Apple ecosystem (Find My, Spatial Audio, etc.)
Improved battery life and charging case design

❌ Cons
Expensive compared to other premium earbuds
Limited customization and features for Android users

Slightly bulky case for smaller pockets
review by: Angel Mainali

 """
)

print(result.keys())
print("Full result:", result)
print("Summary:", result['summary'])
print("Sentiment:", result['sentiment'])
print("Pros:", result['pros'])
print("Cons:", result['cons'])
print("Reviewer:", result['name'])
