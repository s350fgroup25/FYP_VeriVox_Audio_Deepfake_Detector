package hk.omyu.asvonnxtest

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ensureModelInExternalStorage(this)
        setContent { ONNXTestScreen() }
    }

    private fun ensureModelInExternalStorage(context: Context): File {
        val modelDir = context.getExternalFilesDir(null)!!
        val modelFile = File(modelDir, "model.onnx")
        if (!modelFile.exists()) {
            Log.d("ONNXTest", "Copying 1.2GB model to external storage...")
            context.assets.open("model.onnx").use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output, bufferSize = 8 * 1024 * 1024)
                }
            }
            Log.d("ONNXTest", "Model copied: ${modelFile.length() / (1024*1024*1024f)} GB")
        }
        return modelFile
    }
}

@Composable
private fun ONNXTestScreen() {
    val context = LocalContext.current
    var resultText by remember { mutableStateOf("Select an audio file to run inference") }
    var selectedFileName by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    var selectedUri by remember { mutableStateOf<Uri?>(null) }

    // File picker launcher
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            selectedUri = uri
            selectedFileName = getFileName(context, uri)
            resultText = "Selected: $selectedFileName\nPress 'Run Inference' to process"
            Log.d("ONNXTest", "File selected: $selectedFileName")
        }
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        // File selection button
        Button(
            onClick = {
                if (!isLoading) {
                    Log.d("ONNXTest", "Opening file picker")
                    filePickerLauncher.launch("audio/*")
                }
            },
            modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
            enabled = !isLoading
        ) {
            Text("Select Audio File")
        }

        // Show selected file name
        if (selectedFileName != null) {
            Text(
                text = "Selected: $selectedFileName",
                modifier = Modifier.padding(vertical = 4.dp),
                style = MaterialTheme.typography.bodyMedium
            )
        }

        // Run inference button
        Button(
            onClick = {
                if (!isLoading && selectedUri != null) {
                    isLoading = true
                    resultText = "Running inference on $selectedFileName..."
                    Log.d("ONNXTest", "Starting inference on: $selectedFileName")

                    runTestOnnx(context, selectedUri!!) { finalLogits ->
                        isLoading = false
                        val maxLogit = if (finalLogits.any { it.isNaN() }) {
                            Float.NaN
                        } else {
                            finalLogits.maxOrNull() ?: Float.NaN
                        }
                        val argmax = if (!maxLogit.isNaN()) {
                            finalLogits.indexOfFirst { it == maxLogit }
                        } else {
                            -1
                        }
                        val report = """
                            File: $selectedFileName
                            logits length: ${finalLogits.size}
                            max logit: $maxLogit
                            argmax: $argmax
                            ${if (argmax == 0) "✅ Class 0 predicted" else "❌ Class 1 predicted"}
                            ${if (finalLogits.any { it.isNaN() }) "⚠️ NaN detected in output" else ""}
                        """.trimIndent()
                        Log.d("ONNXTest", report)
                        resultText = report
                    }
                } else if (selectedUri == null) {
                    resultText = "Please select an audio file first"
                }
            },
            modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
            enabled = !isLoading && selectedUri != null
        ) {
            Text(if (isLoading) "Processing..." else "Run Inference")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Result display
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Text(
                text = resultText,
                modifier = Modifier.padding(16.dp),
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

private fun getFileName(context: Context, uri: Uri): String {
    var fileName = "unknown"
    context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIndex >= 0) {
                fileName = cursor.getString(nameIndex)
            }
        }
    }
    return fileName
}

private fun runTestOnnx(context: Context, audioUri: Uri, callback: (FloatArray) -> Unit) {
    Thread {
        var env: OrtEnvironment? = null
        var session: OrtSession? = null
        var inputTensor: OnnxTensor? = null

        try {
            Log.d("ONNXTest", "=== START ===")
            env = OrtEnvironment.getEnvironment()
            Log.d("ONNXTest", "✓ Environment OK")

            val modelFile = File(context.getExternalFilesDir(null), "model.onnx")
            Log.d("ONNXTest", "Model file: ${modelFile.absolutePath}, size: ${modelFile.length()}")

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            session = env.createSession(modelFile.absolutePath, sessionOptions)
            Log.d("ONNXTest", "✓ Session created!")

            // Read and process audio file from URI
            val processedFloats = readAndProcessAudio(context, audioUri)

            val targetLength = 77824

            // Ensure correct length
            val finalAudio = if (processedFloats.size == targetLength) {
                processedFloats
            } else {
                FloatArray(targetLength).apply {
                    if (processedFloats.size < targetLength) {
                        System.arraycopy(processedFloats, 0, this, 0, processedFloats.size)
                    } else {
                        System.arraycopy(processedFloats, 0, this, 0, targetLength)
                    }
                }
            }

            // Check input range
            val maxVal = finalAudio.maxOrNull() ?: 0f
            val minVal = finalAudio.minOrNull() ?: 0f
            Log.d("ONNXTest", "Input range: [$minVal, $maxVal]")

            // Robust normalization
            val mean = finalAudio.average().toFloat()
            val variance = finalAudio.map { it - mean }
                .fold(0f) { acc, v -> acc + v * v } / finalAudio.size
            val std = sqrt(variance + 1e-6f)

            if (std.isNaN() || std == 0f) {
                Log.e("ONNXTest", "Invalid std deviation: $std")
                (context as? ComponentActivity)?.runOnUiThread {
                    callback(FloatArray(2))
                }
                return@Thread
            }

            val normalized = FloatArray(finalAudio.size) { i ->
                (finalAudio[i] - mean) / std
            }

            // Verify no NaN values
            if (normalized.any { it.isNaN() }) {
                Log.e("ONNXTest", "NaN values detected in normalized array")
                (context as? ComponentActivity)?.runOnUiThread {
                    callback(FloatArray(2))
                }
                return@Thread
            }

            Log.d("ONNXTest", "✓ Normalized: ${normalized.size}, mean: $mean, std: $std")

            // Create input tensor
            val inputShape = longArrayOf(1, targetLength.toLong())
            val floatBuffer = FloatBuffer.wrap(normalized)
            inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)
            Log.d("ONNXTest", "✓ Input tensor: ${inputShape.joinToString()}")

            // Run inference
            Log.d("ONNXTest", "Running inference...")
            val results = session.run(mapOf("input_values" to inputTensor))

            // Get output
            val outputTensor = results.toList().first().value as? OnnxTensor
                ?: throw IllegalStateException("No output tensor")

            val logits = outputTensor.floatBuffer.array()
            Log.d("ONNXTest", "✓ Inference complete: ${logits.size} logits")

            // Check for NaN in output
            if (logits.any { it.isNaN() }) {
                Log.e("ONNXTest", "Model output contains NaN values")
            } else {
                Log.d("ONNXTest", "Logits: ${logits.joinToString()}")
            }

            results.close()

            // Return results on UI thread
            (context as? ComponentActivity)?.runOnUiThread {
                callback(logits)
            }
            Log.d("ONNXTest", "=== SUCCESS ===")

        } catch (e: Exception) {
            Log.e("ONNXTest", "❌ Failed", e)
            (context as? ComponentActivity)?.runOnUiThread {
                Toast.makeText(context, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                callback(FloatArray(2))
            }
        } finally {
            // Clean up resources
            try { inputTensor?.close() } catch (e: Exception) { Log.w("ONNXTest", "Error closing inputTensor", e) }
            try { session?.close() } catch (e: Exception) { Log.w("ONNXTest", "Error closing session", e) }
            try { env?.close() } catch (e: Exception) { Log.w("ONNXTest", "Error closing env", e) }
        }
    }.start()
}

private fun readAndProcessAudio(context: Context, uri: Uri): FloatArray {
    context.contentResolver.openInputStream(uri)?.use { inputStream ->
        // Try to read as WAV first
        try {
            // Read header to check if it's WAV
            val markSupported = inputStream.markSupported()
            if (markSupported) {
                inputStream.mark(44)
            }

            val header = ByteArray(44)
            val bytesRead = inputStream.read(header)

            if (markSupported) {
                inputStream.reset()
            }

            // Check if it's a WAV file
            if (bytesRead >= 44 && String(header, 0, 4) == "RIFF" && String(header, 8, 4) == "WAVE") {
                Log.d("ONNXTest", "Detected WAV file format")
                return readWavFromStream(inputStream, header)
            }
        } catch (e: Exception) {
            Log.w("ONNXTest", "Error detecting file format, trying as raw audio", e)
        }

        // If not WAV or detection failed, try reading as raw PCM
        Log.d("ONNXTest", "Reading as raw PCM audio")
        return readRawPcmFromStream(inputStream)
    }

    throw IllegalArgumentException("Unable to open audio file")
}

private fun readWavFromStream(inputStream: java.io.InputStream, header: ByteArray): FloatArray {
    // Get audio format info from header
    val numChannels = (header[22].toInt() and 0xFF) or ((header[23].toInt() and 0xFF) shl 8)
    val bitsPerSample = (header[34].toInt() and 0xFF) or ((header[35].toInt() and 0xFF) shl 8)

    Log.d("ONNXTest", "WAV info: channels=$numChannels, bits=$bitsPerSample")

    // Read remaining PCM data
    val pcmData = inputStream.readBytes()

    // Convert to float array (mono by averaging channels if needed)
    val bytesPerSample = bitsPerSample / 8
    val samplesPerChannel = pcmData.size / bytesPerSample / numChannels

    return when (bitsPerSample) {
        16 -> {
            val floatArray = FloatArray(samplesPerChannel)
            val buffer = ByteBuffer.wrap(pcmData).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until samplesPerChannel) {
                var sum = 0
                repeat(numChannels) {
                    if (buffer.hasRemaining()) {
                        sum += buffer.short.toInt()
                    }
                }
                floatArray[i] = (sum / numChannels) / 32768.0f
            }
            floatArray
        }
        32 -> {
            val floatArray = FloatArray(samplesPerChannel)
            val buffer = ByteBuffer.wrap(pcmData).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until samplesPerChannel) {
                var sum = 0f
                repeat(numChannels) {
                    if (buffer.hasRemaining()) {
                        sum += buffer.float
                    }
                }
                floatArray[i] = sum / numChannels
            }
            floatArray
        }
        else -> throw IllegalArgumentException("Unsupported bits per sample: $bitsPerSample")
    }
}

private fun readRawPcmFromStream(inputStream: java.io.InputStream): FloatArray {
    val rawData = inputStream.readBytes()
    val floatArray = FloatArray(rawData.size / 2)
    val buffer = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN)

    for (i in floatArray.indices) {
        if (buffer.hasRemaining()) {
            floatArray[i] = buffer.short / 32768.0f
        }
    }

    return floatArray
}