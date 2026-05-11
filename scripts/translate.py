from deep_translator import GoogleTranslator

translated = GoogleTranslator(
    source="en",
    target="hy",
).translate("apple")

print(translated)
