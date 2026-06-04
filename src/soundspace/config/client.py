from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from soundspace.config.env import ENV_FILE


class SpotifySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPOTIFY_", env_file=ENV_FILE, extra="ignore"
    )

    client_id: SecretStr | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None

    @property
    def configured(self) -> bool:
        return (
            self.client_id is not None
            and self.client_secret is not None
            and self.refresh_token is not None
        )
