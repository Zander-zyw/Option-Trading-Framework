### import DeribitClient ###
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Logger.Logger import logger
### import Logger ###

from datetime import datetime, timezone
import hmac
import hashlib
import base64
import json
import aiohttp
import asyncio
import urllib.parse
class OKXClient:
    def __init__(self, base_url="https://www.okx.com"):
        """
        初始化 OKXClient 实例
        :param api_key: 用户的 API Key
        :param secret_key: 用户的 API Secret
        :param passphrase: 用户的 API Passphrase
        :param base_url: 请求的基础 URL（默认为生产环境地址）
        """
        with open("Account_Info/OKX/API_Key.txt", "r") as f:
            self.api_key = f.read()
        with open("Account_Info/OKX/Secret_Key.txt", "r") as f:
            self.secret_key = f.read()
        with open("Account_Info/OKX/Passphrase.txt", "r") as f:
            self.passphrase = f.read()
        
        # == Initialize Variables ==
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

    # == Get OK-ACCESS-TIMESTAMP ==
    def _get_timestamp(self):
        """
        获取符合 ISO8601 格式的当前 UTC 时间戳，精确到毫秒
        """
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

    # == Get Request Headers ==
    def _get_headers(self, method, request_path, body=""):
        """
        构造请求头，包含签名信息
        :param method: HTTP 请求方法（如 "GET", "POST"）
        :param request_path: API 接口路径，例如 "/api/v5/account/positions"
        :param body: 请求体字符串（如果有）
        :return: 包含签名的 HTTP 请求头
        """
        timestamp = self._get_timestamp()
        message = timestamp + method.upper() + request_path + body
        # 签名：使用 HMAC-SHA256 算法后进行 Base64 编码
        signature = hmac.new(self.secret_key.encode('utf-8'),
                             message.encode('utf-8'),
                             hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature).decode()

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature_b64,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        return headers
    
    def _get_body_str(self, params):
        body_str = ""
        if params:
            body_str += "?"
            for key, value in params.items():
                body_str += f"{key}={value}&"

        return body_str[:-1]

    def _get_full_request_path(self, request_path, params):
        if params:
            query_str = "?" + urllib.parse.urlencode(sorted(params.items()))
            return request_path + query_str
        return request_path

    # == Connect ==
    async def connect(self):
        """
        模拟登录/连接，实际场景下，可以通过获取账户余额来验证 API Key 是否有效
        :return: 布尔值，连接成功返回 True，否则返回 False
        """
        request_path = "/api/v5/account/balance"
        url = self.base_url + request_path
        headers = self._get_headers("GET", request_path)
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                logger.info("Connected successfully!")
                return True
            else:
                error_text = await response.text()
                logger.error(f"Connection failed: {error_text}")
                return False

    # == Get Instruments ==
    async def get_instruments(self, instType: str, uly: str = None, instFamily: str = None):
        """
        获取所有交易对
        :param instType: 交易对类型，可选值为 "SPOT" 或 "MARGIN"或"SWAP"或"FUTURES"或"OPTION"
        :param uly: 标的指数，当 instType 为 OPTION 时，uly或instFamily必填
        :param instFamily: 交易对所属的家族，当 instType 为 OPTION 时，uly或instFamily必填
        :return: 包含所有交易对的 JSON 数据，如果请求失败返回 None
        """

        params = {
            "instType": instType
        }
        if instType == "OPTION":
            if uly:
                params["uly"] = uly
            if instFamily:
                params["instFamily"] = instFamily
            if not uly and not instFamily:
                logger.error("uly or instFamily is required when instType is OPTION")
                return None

        request_path = "/api/v5/account/instruments"
        full_request_path = self._get_full_request_path(request_path, params)

        url = self.base_url + full_request_path

        async with self.session.get(url, headers=self._get_headers("GET", full_request_path)) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"Failed to retrieve instruments: {error_text}")
                return None
    
    # == Get Balance ==
    async def get_balance(self, ccy: str = None):
        """
        获取持仓信息
        :return: JSON 格式的持仓数据，如果请求失败返回 None
        """

        params = {}
        if ccy is not None:
            params["ccy"] = ccy

        request_path = "/api/v5/account/balance"
        full_request_path = self._get_full_request_path(request_path, params)

        url = self.base_url + full_request_path

        async with self.session.get(url, headers=self._get_headers("GET", request_path, self._get_body_str(params))) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"Failed to retrieve positions: {error_text}")
                return None

    # == Disconnect ==
    async def disconnect(self):
        """
        关闭 aiohttp session
        """
        await self.session.close()

    # == Place Order ==
    async def place_order(self, instId: str, side: str, ordType: str, sz: str, px: str = None):
        """
        下单
        """
        


# 示例：使用异步方式调用 OKXClient
async def main():
    client = OKXClient()
    
    # 验证 API Key 是否有效
    if await client.connect():
        response = await client.get_instruments("OPTION", uly="BTC-USD")
        if response:
            print(json.dumps(response, indent=2))
    # 关闭 session
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
