from core.chatbot import ChatBot

if __name__ == '__main__':
    bot = ChatBot()
    while True:
        message = input('ME > ')
        re_text = bot.response(message)
        try:
            print('AI > ' + re_text)
        except:
            print('AI > ' + re_text.decode('utf-8'))