"""Контракт имён и колонок шести семейств 109; только схема, без данных."""

HEAD = {
    "kar": "created_date,updated_date,submission_date,appeal_type,source,region,district,appeal_address,category,sub_category,executor_gov_org,answer_type,type",
    "kos": "incidentid,incidentcode,createddate,servicelevel1,servicelevel2,servicelevel3,sla,finishdate,daysspent,startdate,confirmdate,slabreach,category,openagain,source,status,organizationname,grade,xcoordinate,ycoordinate,region,expireddays,result,updateddate,uploadeddate",
    "tur": "incidentid,incidentcode,createddate,servicelevel1,servicelevel2,servicelevel3,sla,finishdate,daysspent,startdate,confirmdate,slabreach,category,openagain,source,status,organizationname,grade,xcoordinate,ycoordinate,region,expireddays,updateddate,uploadeddate",
    "vko": "application_number,creation_date,closing_date,region,district,street,full_name,applicant_number,application_type,submittal_channel,category,service,contractor,com_exp,result,status,operator",
    "alm": "application_number,creation_date,closing_date,com_exp,result,contractor,status,category,service,status_1,submittal_channel",
    "akm": "request_number,request_subject,creation_date,status,current_project,planned_closing_date,completion_deadline,region_g_a,direction",
    "pav": "id,public_code,create_date,status,service_id,service_name,category_id,category_name,request_type_id,request_type,sdu_load_date",
}

FILES = {
    "kar": "Обращения граждан 109 - Карагандинская область.csv",
    "kos": "Обращения жителей 109 - Костанайская область.csv",
    "tur": "Обращения жителей 109 - Туркестанская область.csv",
    "vko": "Обращения граждан 109 - Восточно-Казахстанская область.csv",
    "alm": "Обращения граждан 109 - Алматинская область.csv",
    "akm": "Обращения граждан 109 - Акмолинская область.csv",
    "pav1": "Обращения граждан Павлодар/Данные по обращениям 109 — Павлодарская область_part_001_of_002.csv",
    "pav2": "Обращения граждан Павлодар/Данные по обращениям 109 — Павлодарская область_part_002_of_002.csv",
}
