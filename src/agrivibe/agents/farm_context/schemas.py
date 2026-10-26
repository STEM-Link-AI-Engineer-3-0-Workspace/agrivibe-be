from typing import Literal

from pydantic import BaseModel, Field


class FieldHealthReport(BaseModel):
    """The Farm Context Agent's answer: a worded summary, plus the satellite numbers it is based on.

    The numbers mirror FieldSat's SatelliteContext, so code can read them without parsing
    the summary. Each one is null when FieldSat did not return it.
    """

    field_id: str = Field(description="The field the report is about, as list_fields or the request names it.")
    available: bool = Field(
        description="True if get_satellite_analysis returned data for this field. False if it failed; the numbers are then null."
    )
    summary: str = Field(
        description=(
            "Two to four plain sentences on how green the field is now compared with before, how uniform it is, "
            "and how reliable the reading is. If data quality is poor, say so first. Never name a cause such as "
            "pests, disease or fertiliser. If the data is not available, say which data is missing and why."
        )
    )
    comparison_days: int | None = Field(description="The length of each compared period, in days, as passed to the tool.")
    current_mean_ndvi: float | None = Field(description="Mean NDVI over the latest period.")
    previous_mean_ndvi: float | None = Field(description="Mean NDVI over the period before it.")
    ndvi_change_percentage: float | None = Field(description="Change in mean NDVI between the two periods, in %.")
    trend: (
        Literal["significant_decline", "moderate_decline", "stable", "moderate_increase", "significant_increase"] | None
    ) = Field(description="FieldSat's label for the NDVI change.")
    ndvi_standard_deviation: float | None = Field(description="How much NDVI varies across the field.")
    field_uniformity: Literal["uniform", "moderately_variable", "uneven"] | None = Field(
        description="FieldSat's label for how evenly green the field is."
    )
    valid_observations: int | None = Field(description="How many clear satellite images the reading is based on.")
    average_cloud_coverage_percentage: float | None = Field(description="Average cloud cover over those images, in %.")
    data_quality: Literal["good", "moderate", "poor"] | None = Field(description="FieldSat's grade for the reading.")
