-- aggiungo la colonna stagione
ALTER TABLE PARTITE_SERIE_A
ADD COLUMN Stagione VARCHAR(20);

-- la stagione va dal 10 agosto al 9 agosto dell'anno dopo
UPDATE PARTITE_SERIE_A
SET Stagione = 
CASE
    WHEN month(Date) < 8  THEN CONCAT(CAST(YEAR(Date) - 1 AS CHAR), '/', CAST(YEAR(Date) AS CHAR))
    WHEN( month(Date) = 8 and day(Date)<10) THEN CONCAT(CAST(YEAR(Date) - 1 AS CHAR), '/', CAST(YEAR(Date) AS CHAR))
    ELSE CONCAT(CAST(YEAR(Date) AS CHAR), '/', CAST(YEAR(Date) + 1 AS CHAR))
END;

select stagione,count(*)
from PARTITE_SERIE_A PSA 
group by stagione;

-- aggiungo la colonna giornata
ALTER TABLE PARTITE_SERIE_A
ADD COLUMN Giornata VARCHAR(20);
