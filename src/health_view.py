"""Data Health UI; consumes aggregates, never raw records or category values."""
import json
import pandas as pd

from src.data_health import load


def health_section(st):
    st.subheader("Качество и свежесть данных")
    current, blocked = load()
    if blocked:
        st.error("BLOCKED — " + blocked)
        return True
    if current is None:
        st.warning("WARNING — нет профиля качества этой сборки. Выполните nazar-build-data.")
        return False
    st.caption(f"Источник: {current['source']} · сборка UTC: {current['built_at']} · "
               f"версия: {current['dataset_id'][:12]}. Схема проверена при сборке, "
               "fingerprint канона совпадает. Сырьё при просмотре страницы не перечитывается.")
    st.markdown("**До какой даты мы видим каждый регион?**")
    freshness, quality, completeness, classes = [], [], [], []
    for region, item in current["regions"].items():
        gaps = item["gaps_ge7"]
        freshness.append({"Регион": region, "Первая дата": item["first_date"],
                          "Последняя дата": item["last_date"], "Дней окна": item["history_days"],
                          "Пропуски ≥7 дней": "; ".join(f"{g['first_date']} — {g['last_date']} ({g['days']} д.)" for g in gaps) or "нет",
                          "Forecast": "доступен" if item["forecast"]["available"] else "не применяется",
                          "Почему": item["forecast"]["reason"]})
        dropped = item["discarded"] or {}
        quality.append({"Регион": region, "Статус": item["status"], "Строк": item["rows"],
                        "Дней с наблюдениями": item["observed_days"], "Самый длинный пропуск, дней": item["longest_gap"],
                        "Пропусков ≥7 дней": len(gaps), "Новых категорий": item["new_categories"],
                        "Отсев сдвига полей": dropped.get("field_shift_dropped"),
                        "Отсев по дате": dropped.get("date_dropped"),
                        "Из них прокси сдвига Акмолы": dropped.get("date_shift_proxy"),
                        "Причины": "; ".join(item["reasons"]) or "Правила предупреждений не сработали"})
        completeness.append({"Регион": region, **{key: round(value * 100, 2) for key, value in item["completeness"].items()}})
        classes.append({"Регион": region, **{key: round(value * 100, 2) for key, value in item["class_shares"].items()}})
    st.dataframe(pd.DataFrame(freshness), hide_index=True, width="stretch")
    st.caption("Даты относятся к выгрузке, а не к сегодняшнему дню. Частота обновления не согласована. "
               "Forecast считает жалобы, не все контакты. Историческое окно Караганды подписано отдельно.")
    st.dataframe(pd.DataFrame(quality), hide_index=True, width="stretch")
    with st.expander("Полнота полей, состав потока и правила качества"):
        st.write("Заполненность полей, %. Пустое опциональное поле означает ограничение наблюдаемости. "
                 "Полнота не доказывает полезность: район может быть константой.")
        st.dataframe(pd.DataFrame(completeness), hide_index=True, width="stretch")
        st.write("Состав всех обращений, %: problem — жалобы; info — справки; system — служебные записи.")
        st.dataframe(pd.DataFrame(classes), hide_index=True, width="stretch")
        st.write("WARNING: пропуск ≥7 дней, новые категории, отсев сдвига/дат, категория <95% или drift выше порога. "
                 "BLOCKED: несовместимый источник/схема/fingerprint или неуспешная сборка. "
                 "Общего балла качества нет. Прокси Акмолы уже входит в отсев по дате.")
    with st.expander("Что изменилось между успешными версиями выгрузки"):
        drift = current["drift"]
        if not drift["available"]:
            st.info(drift["reason"])
        elif not drift["changes"]:
            st.info("Сравниваемые агрегаты не изменились; байты версий различаются.")
        else:
            rows = [{"Регион": row["region"], "Что": row["field"],
                     "Было": json.dumps(row["before"], ensure_ascii=False),
                     "Стало": json.dumps(row["after"], ensure_ascii=False),
                     "Порог предупреждения": row["warning"]} for row in drift["changes"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Пороги: строки ±20%; доли классов/тем и полнота ±10 п.п.; любые новые/исчезнувшие категории; "
                   "сокращение окна ≥20%. Доли в сравнении — от 0 до 1. Это описание изменений, не оценка работы региона.")
    return False
