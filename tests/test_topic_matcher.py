"""Token matching mechanics; no acceptance of REAL baseline changes."""
import unittest

from src.topic_mapping import _match, map_topic, norm


class TopicMatcherTests(unittest.TestCase):
    def test_word_prefix(self):
        for value in ("дорожная", "дорожного", "дорожные", "(дорожная)",
                      "ремонт-дорожного покрытия"):
            self.assertTrue(_match("дорожн", norm(value)), value)
        for key, value in (("дорожн", "железнодорожный"),
                           ("сточн", "Восточной"),
                           ("остановк", "постановке"),
                           ("мост", "делимости")):
            self.assertFalse(_match(key, norm(value)), (key, value))

    def test_modes_and_punctuation(self):
        for value in ("провод", "провод,", "(провод)", "СИП-провод"):
            self.assertTrue(_match("^провод ", norm(value)), value)
        for value in ("проводной", "трубопровод", "проводить"):
            self.assertFalse(_match("^провод ", norm(value)), value)
        self.assertTrue(_match("~дорожн", "железнодорожный"))
        self.assertTrue(_match("газа ", "утечка газа,"))
        self.assertFalse(_match("газа ", "газами"))
        self.assertTrue(_match("нет воды", "нет\tводы"))
        with self.assertRaises(ValueError):
            _match("^ ", "текст")

    def test_known_false_matches(self):
        cases = {
            "Определение делимости и неделимости земельных участков": "прочее",
            "Пассажирский железнодорожный транспорт": "транспорт",
            "Сбор и вывоз твердых бытовых отходов с Восточной части":
                "вывоз мусора и санитария",
            "Выявление бесхозяйных земельных участков и постановка на учет": "прочее",
            "Прорыв трубопровода": "водоснабжение и канализация",
            "Перемонтаж газопровода": "ЖКХ",
            "Путепровод": "дороги",
            "Интернет (проводной, беспроводной)": "связь и телекоммуникации",
        }
        for value, expected in cases.items():
            self.assertEqual(map_topic(value), expected, value)

    def test_compounds_and_manual_decisions(self):
        cases = {
            "Ремонт участков автодороги": "дороги",
            "Подсыпка антигололёдного реагента": "благоустройство и озеленение",
            "Опломбировка водосчетчиков": "ЖКХ",
            "Выдача выписок из медкарты": "социальные вопросы",
            "Оформление опекунства (попечительства) над недееспособным": "социальные вопросы",
            "Опека и попечительство": "социальные вопросы",
            "Безхозяйные сети – постановка на учет": "ЖКХ",
            "Технический паспорт": "жилищный фонд",
            "Экскизные проекты": "прочее",
            # A manual assignment must not become a universal prefix rule.
            "Технический паспорт транспортного средства": "транспорт",
        }
        for value, expected in cases.items():
            self.assertEqual(map_topic(value), expected, value)


if __name__ == "__main__":
    unittest.main()
