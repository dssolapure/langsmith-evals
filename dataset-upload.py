from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

example_inputs = [
    ("What are the opening hours on weekdays?", "Monday to Friday, it’s open from 7:30 AM to 8:00 PM."),
    ("Where is Brew & Bite Café located?", "25 Market Street, Manchester, UK."),
    ("How can I contact the café by phone or email?", "Phone: +44 161 555 2345, Email: hello@brewandbite.co.uk"),
    ("Name three drinks available at the café.", "Espresso, Cappuccino, Latte."),
    ("Which food items are on the menu?", "Croissants, Blueberry Muffins, Sourdough Sandwiches, Vegan Wraps, Gluten-Free Brownies."),
    ("Are there vegan options?", "Yes, vegan options are available on request."),
    ("What are the café’s specials during autumn and summer?", "Autumn: Pumpkin Spice Latte, Summer: Fresh Berry Smoothie."),
    ("What allergens should I be aware of?", "Contains gluten, dairy, nuts, and soy. Gluten-free and vegan alternatives are available."),
    ("What milk options do you have for drinks?", "Oat, Almond, and Soy milk."),
    ("Can I get sugar-free syrup in my coffee?", "Yes, sugar-free syrups are available."),
    ("What is the average price range and payment methods?", "£2 – £8 per item. Payment methods include cash, card, contactless, Apple Pay, and Google Pay."),
    ("What can I get under £5 at Brew & Bite Café?", "Croissants (£2), Blueberry Muffins (£2.20), Sourdough Sandwiches (£4.50), Vegan Wraps (£4.20), Gluten-Free Brownies (£2.80)."),
    ("Are pizzas available at the café?", "We don’t sell pizza."),
    ("Can I order a large sushi platter here?", "I’m sorry, I don’t have enough information about that."),
    ("When is the café open on Sunday, and what drinks or snacks can I get?", "Sunday 9:00 AM – 6:00 PM. Drinks like Latte and Iced Coffee, and snacks like Vegan Wraps and Muffins are available.")
]

client = Client()
# TODO: Fill in dataset id
dataset_id = "ff712422-3a6b-4e8a-9334-4aa3ab19cb5f"

# Prepare inputs and outputs for bulk creation
inputs = [{"Question": input_prompt} for input_prompt, _ in example_inputs]
outputs = [{"Answer": output_answer} for _, output_answer in example_inputs]

client.create_examples(
  inputs=inputs,
  outputs=outputs,
  dataset_id=str(dataset_id),
)