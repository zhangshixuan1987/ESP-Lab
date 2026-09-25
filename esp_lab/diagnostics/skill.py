def compute_skill(stats, forecast, time, obs, climy0, climy1, nlead, monthly=False):
    return stats.compute_skill_seasonal(
        forecast,
        time,
        obs,
        str(climy0),
        str(climy1),
        1,
        nlead,
        resamp=0,
        detrend=True,
        monthly=monthly,
    )

