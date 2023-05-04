from core.chatbot import ChatBot

if __name__ == '__main__':
    bot = ChatBot(model_type="gpt2")
    while True:
        message = input('ME > ')
        re_text = bot.response(message)
        # re_text = bot.response(f'<s>{message}</s></s>')
        try:
            print('AI > ' + re_text)
        except:
            print('AI > ' + re_text.decode('utf-8'))
