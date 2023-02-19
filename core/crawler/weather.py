"""查天气 天气+地名"""
import json
import httpx
import os


def get_weather(Content):
    text = Content
    cacheFileName = './core/crawler/stationID.json'
    if not os.path.isfile(cacheFileName):
        return "城市ID获取失败，换个城市试试吧"
    with open(cacheFileName, "r", encoding='utf-8') as fo:
        ID_Data = json.load(fo)
    if text in ID_Data:
        cityID = ID_Data[text]
        return getWeather(cityID)
    else:
        return "城市ID获取失败，说详细一点试试吧"


def getWeather(cityID):
    url = "http://www.nmc.cn/rest/weather"
    params = {
        "stationid": cityID
    }
    try:
        res = httpx.get(url, params=params).json()
    except Exception as e:
        print(f"天气请求失败\r\n {e}")
        return
    # print(f"天气: {res}")
    info = res
    info_real = info['data']['real']
    info_predict = info['data']['predict']
    real = '{city}\n气温: {temperature}°C  {info}\n体感温度: {feelst}°C\n气温变化: {temperatureDiff}°C \n空气压力: {airpressure}hPa\n湿度: {humidity} %%\n降雨量: {rain}mm\n{direct} {power}\n{publish_time}'.format(
        city=info_real['station']['province'] + info_real['station']['city'],
        temperature=info_real['weather']['temperature'],
        info=info_real['weather']['info'],
        feelst=info_real['weather']['feelst'],
        temperatureDiff=info_real['weather']['temperatureDiff'],
        airpressure=info_real['weather']['airpressure'],
        humidity=info_real['weather']['humidity'],
        rain=info_real['weather']['rain'],
        direct=info_real['wind']['direct'],
        power=info_real['wind']['power'],
        publish_time=info_real['publish_time']
    )
    predict_detail = info_predict['detail']
    predict = '{publish_time}\n白天: {day_temperature}°C {day_info} {day_direct}{day_power}\n夜间: {temperature}°C {info} {direct}{power}'.format(
        publish_time=info_predict['publish_time'],
        day_temperature=predict_detail[1]['day']['weather']['temperature'],
        day_info=predict_detail[1]['day']['weather']['info'],
        day_direct=predict_detail[1]['day']['wind']['direct'],
        day_power=predict_detail[1]['day']['wind']['power'],
        temperature=predict_detail[1]['night']['weather']['temperature'],
        info=predict_detail[1]['night']['weather']['info'],
        direct=predict_detail[1]['night']['wind']['direct'],
        power=predict_detail[1]['night']['wind']['power']
    )
    return real + '\n----------\n' + predict
