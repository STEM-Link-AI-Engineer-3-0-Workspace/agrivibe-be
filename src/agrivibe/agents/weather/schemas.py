from pydantic import BaseModel, Field


class WeatherReport(BaseModel):
    """The Weather Agent's answer: a worded summary, plus the numbers it is based on.

    The numbers mirror FieldSat's WeatherContext, so code can read them without parsing
    the summary. Each one is null when FieldSat did not return it.
    """

    field_id: str = Field(description="The field the report is about, as list_fields or the request names it.")
    available: bool = Field(
        description="True if get_weather_context returned data for this field. False if it failed; the numbers are then null."
    )
    summary: str = Field(
        description=(
            "Two to four plain sentences on the rain: the last 7 days, the last 48 hours and the next 48 hours, "
            "every number with its unit. No advice and no causes. If the data is not available, say which data "
            "is missing and why."
        )
    )
    rainfall_last_7_days_mm: float | None = Field(description="Total rain in the last 7 days, in mm.")
    rainy_days_last_7_days: int | None = Field(description="How many of the last 7 days had rain.")
    rainfall_last_48_hours_mm: float | None = Field(description="Total rain in the last 48 hours, in mm.")
    rainfall_forecast_next_48_hours_mm: float | None = Field(description="Forecast rain for the next 48 hours, in mm.")
    average_humidity_last_7_days: float | None = Field(description="Average relative humidity over the last 7 days, in %.")
    heavy_rain_recent: bool | None = Field(description="Whether heavy rain fell recently, as FieldSat reports it.")
    heavy_rain_expected: bool | None = Field(description="Whether heavy rain is forecast, as FieldSat reports it.")
