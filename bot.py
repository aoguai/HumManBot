from core.chatbot import HumManBot as HumManBot

if __name__ == '__main__':
    bot = HumManBot()
    while True:
        message = input('ME > ')
        re_text = bot.response(message)
        try:
            print('AI > ' + re_text)
        except:
            print('AI > ' + re_text.decode('utf-8'))