from pandera.pandas import DataFrameModel, Field
from pandera.typing.pandas import Series

class InputSchema(DataFrameModel):
    gender: Series[str] = Field(isin=["male", "female"])
    seniorcitizen: Series[int] = Field(isin=[0, 1])
    partner: Series[str] = Field(isin=["yes", "no"])
    dependents: Series[str] = Field(isin=["yes", "no"])
    phoneservice: Series[str] = Field(isin=["yes", "no"])
    multiplelines: Series[str] = Field(isin=["yes", "no", "no_phone_service"])
    internetservice: Series[str] = Field(isin=["dsl", "fiber_optic", "no"])
    onlinesecurity: Series[str] = Field(isin=["no", "yes", "no_internet_service"])
    onlinebackup: Series[str] = Field(isin=["no", "yes", "no_internet_service"])
    deviceprotection: Series[str] = Field(isin=["no", "yes", "no_internet_service"])
    techsupport: Series[str] = Field(isin=["no", "yes", "no_internet_service"])
    streamingtv: Series[str] = Field(isin=["no", "yes", "no_internet_service"])
    streamingmovies: Series[str] = Field(isin=["no", "yes", "no_internet_service"])
    contract: Series[str] = Field(isin=["month-to-month", "one_year", "two_year"])
    paperlessbilling: Series[str] = Field(isin=["yes", "no"])
    paymentmethod: Series[str] = Field(isin=[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ])
    tenure: Series[int] = Field(ge=0, coerce=True)
    monthlycharges: Series[float] = Field(gt=0, lt=1000, coerce=True)
    totalcharges: Series[float] = Field(ge=0, coerce=True)

    class Config:
        strict = True
        coerce = True
