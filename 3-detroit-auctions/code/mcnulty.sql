-- CREATING TABLES FOR McNULTY

-- DLBA Auctions Closed link:
-- https://data.detroitmi.gov/api/views/tgwk-njih/rows.csv

CREATE TABLE AuctionsClosed (
    address VARCHAR NOT NULL,
    parcelID VARCHAR NOT NULL,
    price MONEY NOT NULL,
    closingDate DATE,
    saleStatus VARCHAR,
    buyerStatus VARCHAR,
    purchaserType VARCHAR,
    program VARCHAR,
    councilDistrict INT,
    neighborhood VARCHAR,
    latitude NUMERIC(8,6),
    longitude NUMERIC(8,6),
    location VARCHAR
);

-- Building Permits link
-- https://data.detroitmi.gov/api/views/xw2a-a7tf/rows.csv

CREATE TABLE buildingpermits (
    permitnumber VARCHAR,
    dateissued DATE,
    datecompleted DATE,
    dateexpired DATE,
    siteaddress VARCHAR,
    permitstatus VARCHAR,
    between1 VARCHAR,
    parcelid VARCHAR NOT NULL,
    lotnumber VARCHAR,
    subdivision VARCHAR,
    casetype VARCHAR,
    casedesc TEXT,
    legaluse VARCHAR,
    estimatedcost MONEY,
    parcelsize FLOAT,
    parcelclustersector INT,
    stores FLOAT,
    parcelfloorarea INT,
    parcelgroudarea INT,
    prc_aka_address VARCHAR,
    bldpermittype VARCHAR,
    bldpermitdesc TEXT,
    fdicndesc VARCHAR,
    bldtypeuse VARCHAR,
    residential VARCHAR,
    bldtype_constcode VARCHAR,
    bldzoningdist VARCHAR,
    bldusegroup VARCHAR,
    bldbasement VARCHAR,
    feetype VARCHAR,
    seqno INT,
    csfcreatedby VARCHAR,
    pcfamtdue MONEY,
    owner_lastname VARCHAR,
    owner_firstname VARCHAR,
    owner_address1 VARCHAR,
    owner_address2 VARCHAR,
    owner_city VARCHAR,
    owner_state VARCHAR,
    owner_zip VARCHAR,
    contr_lastname VARCHAR,
    contr_firstname VARCHAR,
    contr_address1 VARCHAR,
    contr_address2 VARCHAR,
    contr_city VARCHAR,
    contr_state VARCHAR,
    contr_zip VARCHAR,
    site_lcn VARCHAR,
    owner_lcn VARCHAR,
    contr_lcn VARCHAR,
    csm_caseno VARCHAR,
    condforapproval VARCHAR,
    permitno_feetype VARCHAR
);

-- Grab filtered table of permits associated with ParcelIDs in auctions

SELECT auctionsclosed.parcelid AS parcelid,
       dateissued,
       estimatedcost,
       parcelsize,
       bldpermittype,
       bldtypeuse,
       owner_city
FROM auctionsclosed
JOIN buildingpermits
ON auctionsclosed.parcelid = buildingpermits.parcelid
WHERE dateissued > '2014-06-01'
; 
