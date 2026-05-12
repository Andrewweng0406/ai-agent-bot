import google.generativeai as genai

# 請填入新生成的 API Key
# 把你的 Key 貼在裡面，前後加點空格也沒關係，.strip() 會處理掉
raw_key = " AIzaSyCTlELcc42qCI9uQifIyrsp8lgWp4cj3Us " 
genai.configure(api_key=raw_key.strip())

try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("你好，請說：測試成功")
    print(response.text)
except Exception as e:
    print(f"診斷結果：{e}")