/**********************************************************************
 * SISAL — FILE 2: ANNUAL NDVI EXTRACTION AT FIXED SAMPLE POINTS
 * (max or median compositing - see COMPOSITE_METHOD below)
 *
 * Companion to sisal_point_selection.js. Uses a FIXED set of points
 * (chosen once in that script, exported to CSV, uploaded here as a
 * Table asset) instead of drawing a fresh random sample every time -
 * so re-running this script never changes which points you're
 * analyzing, only which years/data you extract for them.
 *
 * BEFORE RUNNING - one-time setup:
 *   1. Run sisal_point_selection.js and download Sisal_sample_points.csv
 *      from Google Drive.
 *   2. In the Earth Engine Code Editor, go to the Assets tab (left
 *      panel) -> New -> "CSV file (Table upload)" -> select that CSV.
 *   3. During upload, Earth Engine may offer to set X/Y columns to
 *      auto-generate point geometry - not required either way, since
 *      this script rebuilds point geometry from lon/lat itself.
 *   4. Once the upload finishes, find the asset in the Assets tab,
 *      copy its full asset ID, and paste it into POINTS_ASSET_ID below.
 *
 * After that one-time setup, just re-run this script whenever you want
 * fresh NDVI data - the point set stays fixed.
 *
 * COMPOSITE_METHOD ('max' or 'median'): run this script TWICE, once with
 * each setting, to get two separate CSVs for comparison. 'max' matches
 * Martinez et al. (2024)'s own choice (picked to fight residual haze -
 * Ju & Masek 2016), but a max is highly sensitive to how many candidate
 * images exist per year - and that count rose sharply over the Landsat
 * record (1 satellite in the 1980s-90s vs up to 4 operating at once by
 * the 2020s), which can manufacture an upward NDVI trend that has
 * nothing to do with real vegetation change. 'median' is far less
 * sensitive to sample size and is the more robust default if you're not
 * specifically trying to replicate the paper's exact method.
 *********************************************************************/

var COMPOSITE_METHOD = 'max';  // <-- change to 'median' and re-run for the second dataset

/*====================================================================
 * 0. LOAD THE FIXED POINTS
 *===================================================================*/

var POINTS_ASSET_ID = 'projects/mdp-sisal/assets/Points';  // <-- UPDATE THIS if needed

var samplePointsRaw = ee.FeatureCollection(POINTS_ASSET_ID);

// Rebuild each point as a fresh Feature with explicit Point geometry from
// lon/lat, rather than relying on whatever geometry the CSV upload
// produced (which came through as MultiPoint) - keeps the geometry type
// clean and unambiguous regardless of upload quirks.
var samplePoints = samplePointsRaw.map(function (f) {
  var lon = ee.Number(f.get('lon'));
  var lat = ee.Number(f.get('lat'));
  return ee.Feature(ee.Geometry.Point([lon, lat]), {
    point_id: f.get('point_id'),
    traj: f.get('traj'),
    lon: lon,
    lat: lat
  });
});

print('Total points loaded (should be 150):', samplePoints.size());
print('Composite method for this run:', COMPOSITE_METHOD);

Map.centerObject(samplePoints, 12);
Map.setOptions('SATELLITE');

var CLASS_COLORS = {
  1: '#12301f', 2: '#9fbf6a', 3: '#e31a1c', 4: '#1f78b4', 5: '#ffd92f'
};
var CLASS_LABELS = {
  1: 'Stable interior', 2: 'Stable edge', 3: 'Monotonic loss',
  4: 'Monotonic gain', 5: 'Ambiguous / high turnover'
};
[1, 2, 3, 4, 5].forEach(function (c) {
  var pts = samplePoints.filter(ee.Filter.eq('traj', c));
  Map.addLayer(pts, {color: CLASS_COLORS[c]},
    'Points - class ' + c + ' (' + CLASS_LABELS[c] + ')');
});


/*====================================================================
 * 1. ANNUAL NDVI PER POINT (Landsat, full archive) -> CSV
 *
 * For each fixed sample point, extracts either the maximum or median
 * NDVI value per calendar year (see COMPOSITE_METHOD above) across the
 * full Landsat record (TM/ETM+/OLI merged, cloud-masked). Uses the FULL
 * CALENDAR YEAR rather than Martinez et al.'s May-Aug growing season
 * window, which is specific to North Carolina's temperate seasonality
 * and doesn't transfer to Sisal's tropical wet/dry cycle.
 *===================================================================*/

var GRID_SCALE = 25;  // must match the scale used when points were selected

function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  return image.addBands(opticalBands, null, true);
}

// QA_PIXEL cloud/shadow bits are defined the same way across all
// Collection 2 Level-2 Landsat sensors.
function maskLandsatClouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudBit = 1 << 3;
  var shadowBit = 1 << 4;
  var mask = qa.bitwiseAnd(cloudBit).eq(0).and(qa.bitwiseAnd(shadowBit).eq(0));
  return image.updateMask(mask);
}

// TM (Landsat 5) and ETM+ (Landsat 7): red = SR_B3, NIR = SR_B4
function addNDVI_TM(image) {
  var ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI');
  return image.addBands(ndvi);
}

// OLI (Landsat 8/9): red = SR_B4, NIR = SR_B5
function addNDVI_OLI(image) {
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  return image.addBands(ndvi);
}

// Filtered by the fixed points' own extent, not a separately-maintained
// AOI polygon - one less thing to keep in sync between the two scripts.
var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(samplePoints);
var l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(samplePoints);
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(samplePoints);
var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(samplePoints);

var landsatTM = l5.merge(l7)
  .map(applyScaleFactors).map(maskLandsatClouds).map(addNDVI_TM).select('NDVI');
var landsatOLI = l8.merge(l9)
  .map(applyScaleFactors).map(maskLandsatClouds).map(addNDVI_OLI).select('NDVI');
var landsatAll = landsatTM.merge(landsatOLI);

print('Total Landsat images over sample points (all sensors, full archive):',
  landsatAll.size());

// One composite per calendar year (max or median, per COMPOSITE_METHOD),
// spanning the full record to the present.
var currentYear = ee.Number(ee.Date(Date.now()).get('year'));
var years = ee.List.sequence(1985, currentYear);

var annualNDVI = ee.ImageCollection(years.map(function (y) {
  y = ee.Number(y);
  var yearImages = landsatAll.filter(ee.Filter.calendarRange(y, y, 'year'));
  var count = yearImages.size();

  // ee.ImageCollection([]).max()/.median() on an EMPTY collection returns
  // an image with NO BANDS at all (not a masked 'NDVI' band) - so a year
  // with zero Landsat images crashes the later .select('NDVI') call
  // downstream. Substitute a fully-masked placeholder that still has a
  // properly named 'NDVI' band for those years, so .select() never
  // fails; reduceRegions then correctly returns null for that
  // point-year, which the notNull() filter below already drops.
  var placeholder = ee.Image.constant(0).rename('NDVI').selfMask();
  var composite = (COMPOSITE_METHOD === 'median') ? yearImages.median() : yearImages.max();
  var compositeImg = ee.Image(ee.Algorithms.If(count.gt(0), composite, placeholder));

  return compositeImg.set('year', y).set('image_count', count);
}));

// Extract the annual composite NDVI at each point, for every year -> long
// table. NOTE: reduceRegions() names its output property after the
// REDUCER used (here, 'mean'), not the band name - renaming the band
// beforehand does NOT change the output column name. The fix is to
// rename the OUTPUT property after reduceRegions runs, which is what
// actually makes the exported column say what the value is.
var ndviColumnName = 'ndvi_' + COMPOSITE_METHOD;  // 'ndvi_max' or 'ndvi_median'

var ndviLongTable = annualNDVI.map(function (image) {
  var year = image.get('year');
  var count = image.get('image_count');
  return image.select('NDVI').reduceRegions({
    collection: samplePoints,
    reducer: ee.Reducer.mean(),
    scale: GRID_SCALE
  }).map(function (f) {
    return f.set('year', year)
            .set('n_images_that_year', count)
            .set(ndviColumnName, f.get('mean'));
  });
}).flatten();

// Drop (point, year) rows with no cloud-free Landsat coverage that year.
ndviLongTable = ndviLongTable.filter(ee.Filter.notNull([ndviColumnName]));

print('Total (point x year) rows in export:', ndviLongTable.size());

Export.table.toDrive({
  collection: ndviLongTable,
  description: 'Sisal_Landsat_annual_' + COMPOSITE_METHOD + '_NDVI_per_point',
  folder: 'GEE_exports',
  fileFormat: 'CSV',
  selectors: ['point_id', 'traj', 'lon', 'lat', 'year', ndviColumnName, 'n_images_that_year']
});

// Resulting CSV columns:
//   point_id           - stable per-point ID (from the point-selection
//                         script's export - the SAME point_id every
//                         year, since it was set explicitly before this
//                         script's per-year .flatten() runs)
//   traj               - trajectory class (1-5) this point was drawn from
//   lon, lat           - point coordinates (WGS84 degrees)
//   year               - calendar year
//   ndvi_max/ndvi_median - composite NDVI at that point for that year
//                         (column name depends on COMPOSITE_METHOD)
//   n_images_that_year - how many cloud-masked Landsat images went into
//                         that year's composite
//
// Download this CSV. In the Python notebook, CONFIG["input_csv"] just
// needs to point at whichever file you're using - the loader reads
// whichever of ndvi_max/ndvi_median is present in the CSV automatically.
