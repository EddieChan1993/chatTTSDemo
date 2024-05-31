import ChatTTS
import soundfile as sf

model_dir = "model/chatTTS"

chat = ChatTTS.Chat()
chat.load_models(source='local', local_path=model_dir)

texts = ["夏夏，你是一头猪猪", ]

wavs = chat.infer(texts, use_decoder=True)
# Audio(wavs[0], rate=24_000, autoplay=True)
sf.write("./radio/demo.wav", wavs[0][0], 24000)
