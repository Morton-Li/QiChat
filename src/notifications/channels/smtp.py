import smtplib
import ssl
from email.headerregistry import Address
from email.message import EmailMessage

from .base import NotificationChannel


class SmtpChannel(NotificationChannel):
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        to_addresses: list[str],
        use_tls: bool = True,
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.mail_from = Address(
            display_name='QiChat Trainer',
            username=username.split('@')[0],
            domain=username.split('@')[1],
        )
        self.to_addresses = to_addresses
        self.use_tls = use_tls

    def send(
        self,
        subject: str,
        content: str,
    ) -> None:
        msg = EmailMessage()
        msg["From"] = self.mail_from
        msg["To"] = ", ".join(self.to_addresses)
        msg["Subject"] = subject
        msg.set_content(content)

        ctx = ssl.create_default_context()
        if self.use_tls:
            with smtplib.SMTP_SSL(host=self.host, port=self.port, timeout=12, context=ctx) as server:
                server.login(user=self.username, password=self.password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host=self.host, port=self.port) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.ehlo()
                server.login(user=self.username, password=self.password)
                server.send_message(msg)
