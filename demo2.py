###################################
# Sample a speaker from Gaussian.
import torch
import ChatTTS
import soundfile as sf

model_dir = "model/chatTTS"
chat = ChatTTS.Chat()
chat.load_models(source='local', local_path=model_dir)

std, mean = torch.load(model_dir + '/asset/spk_stat.pt').chunk(2)
rand_spk = torch.randn(768) * std + mean
params_infer_code = {
    'spk_emb': rand_spk,  # add sampled speaker
    'temperature': .3,  # using custom temperature
    'top_P': 0.7,  # top P decode
    'top_K': 20,  # top K decode
}
# uv_break 语气停顿自然，有嗯语气
# lbreak 停顿干脆
# laugh 笑
params_refine_text = {
    'prompt': '[oral_2][laugh_0][break_4]'
}

# text = " [laugh][uv_break]WTT 粉丝，你好[uv_break]，下班路上[uv_break]，注意安全[uv_break]，别骑快了[uv_break]，注意防晒[uv_break]"
# text = " [laugh][lbreak]我是拿来装我平时网上下的电视剧[uv_break]或者电影的？"
# text = " [laugh][lbreak]我也想玩黑神话悟空[uv_break]，期待的很[uv_break]，但是我没得PC电脑[uv_break]"
text = "马海龙小伙[laugh][lbreak]，在干嘛[uv_break]，你听到的这段语音[uv_break]，是我文字生成的[uv_break]，目前开源的一个AI项目[uv_break]，我正在测试中"
wav = chat.infer(text, params_refine_text=params_refine_text)
# Audio(wavs[0], rate=24_000, autoplay=True)
sf.write("./radio/demo2.wav", wav[0][0], 24000)
