from pyobs.modules import Module

from pyobs_cloudcover.web_api.server import Server


class Application(Module):
    def __init__(self) -> None:
        super(Application, self).__init__()

        #self._server = Server()

    async def open(self) -> None:
        await super(Application, self).open()
        #await self._server.start()
