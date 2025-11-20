from google import genai

# Generate your key here : https://aistudio.google.com/app/api-keys
client = genai.Client(api_key="Your API KEY")

example_instructions = {
    "COCO_train2014_000000318556.jpg": "Use appropriate colors to match: A blue and white bathroom with butterfly themed wall tiles.",
    "COCO_train2014_000000116100.jpg": "Use appropriate colors to match: A wide angle view of the kitchen work area",
    "COCO_train2014_000000379340.jpg": "Make the image colorful as described: A graffiti-ed stop sign across the street from a red car ",
    "COCO_train2014_000000134754.jpg": "Apply realistic colors according to: Two teenagers at a white sanded beach with surfboards.",
    "COCO_train2014_000000538480.jpg": "Make the image colorful as described: a bathroom with toilet and sink and blue wall",
    "COCO_train2014_000000476220.jpg": "Apply realistic colors according to: A small kitchen with several appliances and cookware.",
    "COCO_train2014_000000299675.jpg": "Color the objects to reflect this scene: A kitchen with a tile floor has cabinets with no doors, a dishwasher, a sink, and a refrigerator.",
    "COCO_train2014_000000032275.jpg": "Make the image colorful as described: A clean restroom with towels and washcloths laid out.",
    "COCO_train2014_000000302443.jpg": "Apply realistic colors according to: Silver balls on sand with people walking around. ",
    "COCO_train2014_000000025470.jpg": "Colorize this image so that A kitchen with brown cabinets, tile backsplash, and grey counters.",
    "COCO_train2014_000000513461.jpg": "Colorize this image so that A man getting ready to surf as lookers walk by",
    "COCO_train2014_000000018691.jpg": "Apply realistic colors according to: A group of people are riding the bus.",
    "COCO_train2014_000000285579.jpg": "Colorize this image so that A man walking in the rain crossing a street while holding an umbrella.",
    "COCO_train2014_000000266366.jpg": "Colorize this image so that A clean, mediocre motel bathroom with a nice sink.",
    "COCO_train2014_000000226658.jpg": "Color the objects to reflect this scene: A woman having a slice of bread with jelly.",
}

new_instructions: dict[str, list[str]] = {}

for key, instruction in example_instructions.items():
    prompt = f"""The below instruction in triple backticks are used in a NN pipeline that colorize grayscale images. From the given instruction, generate 5 different versions of it the conveys different vibrance
```
{instruction}
```

The output must only comprise of the new instructions on each new line without any marks such as backticks.
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-001", contents=prompt
    )
    new_instructions[key] = response.text.split("\n")

print(new_instructions.values())
