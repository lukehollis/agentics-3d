using UnityEngine;
using UnityEditor;
using System.IO;
using Newtonsoft.Json.Linq;
using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Unity.EditorCoroutines.Editor;

public class GeoJsonImporter : EditorWindow
{
    private Vector2 knownUnityPoint = new Vector2(-130.6f, -61.7f); // Your Unity reference point (x,z)
    private Vector2 knownLatLon = new Vector2(37.7955956f, -122.3932602f); // Your known lat/lon
    private Vector2 originUTM;
    private string geoDataPath;
    
    // Add these new fields
    private GameObject busStopPrefab;
    private GameObject bartStationPrefab;

    // WGS84 ellipsoid constants
    private const float a = 6378137.0f; // semi-major axis
    private const float e = 0.081819190842622f; // More precise eccentricity
    private const float k0 = 0.9996f; // scale factor

    private void OnEnable()
    {
        // Calculate the UTM coordinates for the known point
        var knownUTM = LatLonToUTM(knownLatLon.x, knownLatLon.y);
        
        // Calculate the origin UTM (where Unity 0,0,0 should be)
        originUTM = new Vector2(
            knownUTM.x - knownUnityPoint.x,
            knownUTM.y - knownUnityPoint.y
        );

        // Test the known point conversion
        var testLatLon = new Vector2(37.7955956f, -122.3932602f);
        var utmResult = LatLonToUTM(testLatLon.x, testLatLon.y);
        Debug.Log($"Test Point Conversion:");
        Debug.Log($"Input Lat/Lon: {testLatLon.x}, {testLatLon.y}");
        Debug.Log($"UTM Result: Easting: {utmResult.x:F2}m, Northing: {utmResult.y:F2}m");
    }

    private Vector2 LatLonToUTM(float lat, float lon)
    {
        // Convert to radians
        float latRad = lat * Mathf.PI / 180.0f;
        float lonRad = lon * Mathf.PI / 180.0f;

        // Get UTM zone and central meridian
        int zone = (int)((lon + 180) / 6) + 1;
        float lonOrigin = ((zone - 1) * 6 - 180 + 3) * Mathf.Deg2Rad;

        // UTM calculations
        float N = a / Mathf.Sqrt(1 - e * e * Mathf.Sin(latRad) * Mathf.Sin(latRad));
        float T = Mathf.Tan(latRad) * Mathf.Tan(latRad);
        float C = e * e * Mathf.Cos(latRad) * Mathf.Cos(latRad) / (1 - e * e);
        float A = Mathf.Cos(latRad) * (lonRad - lonOrigin);

        // Calculate easting and northing
        float M = a * ((1 - e * e / 4 - 3 * e * e * e * e / 64) * latRad
                     - (3 * e * e / 8 + 3 * e * e * e * e / 32) * Mathf.Sin(2 * latRad)
                     + (15 * e * e * e * e / 256) * Mathf.Sin(4 * latRad));

        float easting = k0 * N * (A + (1 - T + C) * A * A * A / 6
                               + (5 - 18 * T + T * T + 72 * C - 58) * A * A * A * A * A / 120)
                               + 500000.0f;

        float northing = k0 * (M + N * Mathf.Tan(latRad) * (A * A / 2 
                                + (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24
                                + (61 - 58 * T + T * T + 600 * C - 330) * A * A * A * A * A * A / 720));

        // Add offset for southern hemisphere
        if (lat < 0)
            northing += 10000000.0f;

        return new Vector2(easting, northing);
    }

    private Vector3 ConvertToUnityPosition(float longitude, float latitude)
    {
        var utm = LatLonToUTM(latitude, longitude);
        
        return new Vector3(
            utm.x - originUTM.x,
            0, // Height can be adjusted if needed
            utm.y - originUTM.y
        );
    }

    [MenuItem("Tools/BART/GeoJSON Importer")]
    public static void ShowWindow()
    {
        GetWindow<GeoJsonImporter>("GeoJSON Importer");
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("BART GeoJSON Importer", EditorStyles.boldLabel);
        
        // File selection
        EditorGUILayout.LabelField("Data Source", EditorStyles.boldLabel);
        using (new EditorGUILayout.VerticalScope("box"))
        {
            geoDataPath = EditorGUILayout.TextField("GeoData Path", geoDataPath);
            if (GUILayout.Button("Browse..."))
            {
                string path = EditorUtility.OpenFolderPanel("Select GeoData Folder", "", "");
                if (!string.IsNullOrEmpty(path))
                {
                    geoDataPath = path;
                }
            }
        }

        EditorGUILayout.Space();

        // Prefab configuration
        EditorGUILayout.LabelField("Prefab Configuration", EditorStyles.boldLabel);
        using (new EditorGUILayout.VerticalScope("box"))
        {
            busStopPrefab = (GameObject)EditorGUILayout.ObjectField(
                "Bus Stop Prefab", 
                busStopPrefab, 
                typeof(GameObject), 
                false
            );
            
            bartStationPrefab = (GameObject)EditorGUILayout.ObjectField(
                "BART Station Prefab", 
                bartStationPrefab, 
                typeof(GameObject), 
                false
            );
        }

        EditorGUILayout.Space();

        // Import buttons
        EditorGUILayout.LabelField("Import Actions", EditorStyles.boldLabel);
        using (new EditorGUILayout.VerticalScope("box"))
        {
            if (GUILayout.Button("Import Bus Stops"))
            {
                ImportBusStopsOnly();
            }
            if (GUILayout.Button("Import BART Stations and Lines"))
            {
                ImportBartOnly();
            }
            if (GUILayout.Button("Import Streets"))
            {
                ImportStreetsOnly();
            }
        }
    }

    private void ImportBusStopsOnly()
    {
        if (string.IsNullOrEmpty(geoDataPath))
        {
            EditorUtility.DisplayDialog("Error", "Please set the GeoData path", "OK");
            return;
        }

        if (busStopPrefab == null)
        {
            EditorUtility.DisplayDialog("Error", "Please assign a Bus Stop prefab", "OK");
            return;
        }

        GameObject stopsParent = new GameObject("Bus_Stops");
        try
        {
            if (File.Exists(Path.Combine(geoDataPath, "muni_stops.geojson")))
            {
                ImportBusStops(Path.Combine(geoDataPath, "muni_stops.geojson"), stopsParent.transform);
                EditorUtility.DisplayDialog("Success", "Bus stops imported successfully!", "OK");
            }
        }
        catch (System.Exception e)
        {
            EditorUtility.DisplayDialog("Error", $"Import failed: {e.Message}", "OK");
            Debug.LogException(e);
            DestroyImmediate(stopsParent);
        }
    }

    private void ImportBartOnly()
    {
        if (string.IsNullOrEmpty(geoDataPath))
        {
            EditorUtility.DisplayDialog("Error", "Please set the GeoData path", "OK");
            return;
        }

        if (bartStationPrefab == null)
        {
            EditorUtility.DisplayDialog("Error", "Please assign a BART Station prefab", "OK");
            return;
        }

        GameObject bartParent = new GameObject("BART_System");
        try
        {
            if (File.Exists(Path.Combine(geoDataPath, "bart_lines.geojson")))
            {
                ImportBartLines(Path.Combine(geoDataPath, "bart_lines.geojson"), bartParent.transform);
                EditorUtility.DisplayDialog("Success", "BART system imported successfully!", "OK");
            }
        }
        catch (System.Exception e)
        {
            EditorUtility.DisplayDialog("Error", $"Import failed: {e.Message}", "OK");
            Debug.LogException(e);
            DestroyImmediate(bartParent);
        }
    }

    private void ImportStreetsOnly()
    {
        if (string.IsNullOrEmpty(geoDataPath))
        {
            EditorUtility.DisplayDialog("Error", "Please set the GeoData path", "OK");
            return;
        }

        string filePath = Path.Combine(geoDataPath, "bay_area_major_streets.geojson");
        Debug.Log($"Attempting to import streets from: {filePath}");
        
        if (!File.Exists(filePath))
        {
            EditorUtility.DisplayDialog("Error", "streets.geojson not found", "OK");
            return;
        }

        EditorCoroutineUtility.StartCoroutine(ImportStreetsCoroutine(filePath), this);
    }

    private System.Collections.IEnumerator ImportStreetsCoroutine(string filePath)
    {
        GameObject streetsParent = new GameObject("Streets");
        Debug.Log("Reading file...");
        string jsonContent = File.ReadAllText(filePath);
        Debug.Log($"File read complete. Size: {jsonContent.Length / 1024 / 1024}MB");
        
        Debug.Log("Parsing JSON...");
        JObject geoJson;
        try
        {
            geoJson = JObject.Parse(jsonContent);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to parse JSON:");
            Debug.LogException(e);
            EditorUtility.DisplayDialog("Error", $"Import failed: {e.Message}", "OK");
            DestroyImmediate(streetsParent);
            yield break;
        }

        var features = geoJson["features"] as JArray;
        if (features == null)
        {
            Debug.LogError("No features found in GeoJSON");
            DestroyImmediate(streetsParent);
            yield break;
        }

        int totalFeatures = features.Count;
        Debug.Log($"Found {totalFeatures} total features");
        
        const int batchSize = 100;
        int processedFeatures = 0;
        int createdRoads = 0;
        
        // Process in batches
        for (int i = 0; i < totalFeatures; i += batchSize)
        {
            int currentBatchSize = Mathf.Min(batchSize, totalFeatures - i);
            Debug.Log($"Processing batch {i/batchSize + 1}, features {i} to {i + currentBatchSize}");
            
            var batch = features.Skip(i).Take(currentBatchSize);
            foreach (var feature in batch)
            {
                processedFeatures++;
                string geometryType = feature["geometry"]["type"].ToString();
                
                if (geometryType == "LineString" || geometryType == "MultiLineString")
                {
                    string streetName = feature["properties"]["name"]?.ToString() ?? "Unnamed Street";
                    GameObject roadObj = new GameObject(streetName);
                    roadObj.transform.parent = streetsParent.transform;
                    
                    Road road = roadObj.AddComponent<Road>();
                    road.numLanes = 2;
                    
                    if (geometryType == "LineString")
                    {
                        var coordinates = feature["geometry"]["coordinates"];
                        foreach (var coord in coordinates)
                        {
                            GameObject node = new GameObject("Node");
                            node.transform.parent = roadObj.transform;
                            
                            float longitude = coord[0].Value<float>();
                            float latitude = coord[1].Value<float>();
                            node.transform.position = ConvertToUnityPosition(longitude, latitude);
                            
                            road.nodes.Add(node.transform);
                        }
                    }
                    else // MultiLineString
                    {
                        var lineStrings = feature["geometry"]["coordinates"];
                        foreach (var coord in lineStrings[0]) // Take first line string
                        {
                            GameObject node = new GameObject("Node");
                            node.transform.parent = roadObj.transform;
                            
                            float longitude = coord[0].Value<float>();
                            float latitude = coord[1].Value<float>();
                            node.transform.position = ConvertToUnityPosition(longitude, latitude);
                            
                            road.nodes.Add(node.transform);
                        }
                    }
                    createdRoads++;
                }
            }
            
            // Force garbage collection after each batch
            System.GC.Collect();
            Debug.Log($"Batch complete. Total progress: {processedFeatures}/{totalFeatures} features, {createdRoads} roads created");
            
            EditorUtility.DisplayProgressBar("Importing Streets", 
                $"Processed {processedFeatures}/{totalFeatures} features...", 
                (float)processedFeatures / totalFeatures);
            
            // Yield to let the editor breathe and update
            yield return new EditorWaitForSeconds(0.1f);
            
            // Force garbage collection after each batch
            System.GC.Collect();
            Debug.Log($"Batch complete. Total progress: {processedFeatures}/{totalFeatures} features, {createdRoads} roads created");
            
            EditorUtility.DisplayProgressBar("Importing Streets", 
                $"Processed {processedFeatures}/{totalFeatures} features...", 
                (float)processedFeatures / totalFeatures);
            
            // Yield to let the editor breathe and update
            yield return new EditorWaitForSeconds(0.1f);
        }
        
        EditorUtility.ClearProgressBar();
        Debug.Log($"Import complete! Processed {processedFeatures} features and created {createdRoads} roads");
        EditorUtility.DisplayDialog("Success", $"Streets imported successfully! Created {createdRoads} roads", "OK");
    }

    private void ImportBusStops(string filePath, Transform parent)
    {
        string jsonContent = File.ReadAllText(filePath);
        JObject geoJson = JObject.Parse(jsonContent);

        if (busStopPrefab == null)
        {
            Debug.LogError("Bus Stop prefab not assigned!");
            return;
        }

        foreach (var feature in geoJson["features"])
        {
            if (feature["geometry"]["type"].ToString() == "Point")
            {
                var coordinates = feature["geometry"]["coordinates"];
                float longitude = coordinates[0].Value<float>();
                float latitude = coordinates[1].Value<float>();
                
                // Use the assigned prefab directly
                GameObject stopInstance = PrefabUtility.InstantiatePrefab(busStopPrefab, parent) as GameObject;
                stopInstance.name = feature["properties"]["stop_name"]?.ToString() ?? "Bus Stop";
                stopInstance.transform.position = ConvertToUnityPosition(longitude, latitude);
            }
        }
    }

    private void ImportBartLines(string filePath, Transform parent)
    {
        string jsonContent = File.ReadAllText(filePath);
        JObject geoJson = JObject.Parse(jsonContent);

        if (bartStationPrefab == null)
        {
            Debug.LogError("BART Station prefab not assigned!");
            return;
        }

        foreach (var feature in geoJson["features"])
        {
            if (feature["geometry"]["type"].ToString() == "LineString")
            {
                // Create RailTrack for each line
                GameObject trackObj = new GameObject($"BART_Line_{feature["properties"]["line_name"]}");
                trackObj.transform.parent = parent;
                
                RailTrack track = trackObj.AddComponent<RailTrack>();
                
                // Set track color from properties
                string colorHex = feature["properties"]["color"]?.ToString() ?? "#FFFFFF";
                if (ColorUtility.TryParseHtmlString(colorHex, out Color trackColor))
                {
                    track.lineColor = trackColor;
                }
                
                // Add nodes along the track
                var coordinates = feature["geometry"]["coordinates"];
                foreach (var coord in coordinates)
                {
                    GameObject node = new GameObject("Node");
                    node.transform.parent = trackObj.transform;
                    
                    float longitude = coord[0].Value<float>();
                    float latitude = coord[1].Value<float>();
                    node.transform.position = ConvertToUnityPosition(longitude, latitude);
                    
                    track.nodes.Add(node.transform);
                }
            }
            else if (feature["geometry"]["type"].ToString() == "Point")
            {
                // Create BART station using the assigned prefab
                var coordinates = feature["geometry"]["coordinates"];
                float longitude = coordinates[0].Value<float>();
                float latitude = coordinates[1].Value<float>();
                
                GameObject stationInstance = PrefabUtility.InstantiatePrefab(bartStationPrefab, parent) as GameObject;
                stationInstance.name = feature["properties"]["station_name"]?.ToString() ?? "BART Station";
                stationInstance.transform.position = ConvertToUnityPosition(longitude, latitude);
            }
        }
    }

}
