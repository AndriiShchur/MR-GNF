# download_era5_europe_0p25.py
import cdsapi
import calendar
import os

c = cdsapi.Client()

def download_era5_europe(years, output_dir="./data/raw/025deg/"):
    """
    Download ERA5 reanalysis for the NE Atlantic / Western Europe box
    (lat 40–65 ° N, lon –18–8 °) at 0.25 °×0.25 ° resolution, month-by-month.

    Parameters
    ----------
    years : list[int] | list[str]
        Years to fetch, e.g. [1980, 1981, …].
    output_dir : str, optional
        Folder root where sub-folders `surface/` and `pressure_levels/`
        will be created automatically.
    """
    # -----------------------------------------------------------------
    # Folder layout
    # -----------------------------------------------------------------
    surf_dir   = os.path.join(output_dir, "surface")
    press_dir  = os.path.join(output_dir, "pressure_levels")
    os.makedirs(surf_dir,  exist_ok=True)
    os.makedirs(press_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Request templates
    # -----------------------------------------------------------------
    # NOTE: CDS expects [N, W, S, E]  ➜  [lat_max, lon_min, lat_min, lon_max]
    # area = [61, -13, 47, 4]          # Europe / GB + nearby 
    area = [75, -50, 30, 20]
    grid = [0.25, 0.25]              # 0.25-degree cells
    times = ["00:00", "06:00", "12:00", "18:00"]

    surface_vars = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    surface_precip = ["total_precipitation"]

    pressure_vars = [
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "relative_humidity",
        "geopotential",
    ]
    pressure_levels = ["850", "500", "300"]   # hPa

    # -----------------------------------------------------------------
    # Iterate year → month   (keeps each request < ~1 GB)
    # -----------------------------------------------------------------
    for year in years:
        y = int(year)
        for m in range(1, 13):
            month = f"{m:02d}"
            days = [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]

            # ---------- single-level (surface) variables ----------
            surf_out = os.path.join(surf_dir, f"surface_{y}_{month}.nc")
            if not os.path.exists(surf_out):
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "format":      "netcdf",
                        "variable":    surface_vars,
                        "year":        str(y),
                        "month":       month,
                        "day":         days,
                        "time":        times,
                        "area":        area,
                        "grid":        grid,
                    },
                    surf_out,
                )

            # ---------- precipitation (kept separate) -------------
            precip_out = os.path.join(surf_dir, f"surface_p_{y}_{month}.nc")
            if not os.path.exists(precip_out):
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "format":      "netcdf",
                        "variable":    surface_precip,
                        "year":        str(y),
                        "month":       month,
                        "day":         days,
                        "time":        times,
                        "area":        area,
                        "grid":        grid,
                    },
                    precip_out,
                )

            # ---------- pressure-level variables ------------------
            press_out = os.path.join(press_dir, f"pressure_{y}_{month}.nc")
            if not os.path.exists(press_out):
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type":  "reanalysis",
                        "format":        "netcdf",
                        "variable":      pressure_vars,
                        "pressure_level": pressure_levels,
                        "year":          str(y),
                        "month":         month,
                        "day":           days,
                        "time":          times,
                        "area":          area,
                        "grid":          grid,
                    },
                    press_out,
                )


if __name__ == "__main__":
    # Example: 1980-2024 inclusive
    years = range(1980, 2025)
    download_era5_europe(years)