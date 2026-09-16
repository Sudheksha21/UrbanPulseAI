def analyze_causes(
    traffic,
    hour,
    day_of_week,
    rain,
    snow,
    clouds
):

    causes = []


    # Traffic level

    if traffic >= 4500:

        causes.append(
            "High predicted traffic volume"
        )


    elif traffic >= 2500:

        causes.append(
            "Moderate traffic conditions"
        )


    # Rush hour

    if hour in [7, 8, 9]:

        causes.append(
            "Morning peak-hour traffic"
        )


    if hour in [16, 17, 18, 19]:

        causes.append(
            "Evening peak-hour traffic"
        )


    # Weather

    if rain > 0:

        causes.append(
            "Rain may be affecting traffic conditions"
        )


    if snow > 0:

        causes.append(
            "Snow may be affecting traffic conditions"
        )


    if clouds >= 70:

        causes.append(
            "Heavy cloud coverage"
        )


    # Weekday/weekend

    if day_of_week < 5:

        causes.append(
            "Weekday traffic pattern"
        )

    else:

        causes.append(
            "Weekend traffic pattern"
        )


    # Default

    if not causes:

        causes.append(
            "Normal traffic conditions"
        )


    return causes