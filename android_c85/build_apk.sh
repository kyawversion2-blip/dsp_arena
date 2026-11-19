#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== C85 Android App - APK Build Helper ==="

# 1. Check Java
if ! command -v java &>/dev/null; then
    echo "❌ Java not found. Please install Java 17:"
    echo "   brew install openjdk@17"
    exit 1
fi
echo "✅ Java: $(java -version 2>&1 | head -n 1)"

# 2. Check Android SDK
if [ -z "$ANDROID_HOME" ]; then
    if [ -d "$HOME/Library/Android/sdk" ]; then
        export ANDROID_HOME="$HOME/Library/Android/sdk"
    elif [ -d "/opt/homebrew/share/android-commandlinetools" ]; then
        export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
    fi
fi

if [ -z "$ANDROID_HOME" ] || [ ! -d "$ANDROID_HOME" ]; then
    echo "⚠️ Android SDK not found."
    echo ""
    echo "To build locally via Terminal, install Android Command Line Tools & Gradle:"
    echo "  brew install gradle"
    echo "  brew install --cask android-commandlinetools"
    echo "  yes | sdkmanager --licenses"
    echo "  sdkmanager \"platform-tools\" \"platforms;android-34\" \"build-tools;34.0.0\""
    echo ""
    echo "Alternatively, open this folder in Android Studio:"
    echo "  open -a \"Android Studio\" \"$DIR\""
    exit 1
fi

echo "✅ Android SDK: $ANDROID_HOME"

# 3. Check Gradle
if command -v gradle &>/dev/null; then
    GRADLE_CMD="gradle"
elif [ -f "./gradlew" ]; then
    GRADLE_CMD="./gradlew"
else
    echo "❌ Gradle not found. Please run: brew install gradle"
    exit 1
fi

echo "🚀 Building Debug APK with $GRADLE_CMD..."
$GRADLE_CMD assembleDebug

APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
if [ -f "$APK_PATH" ]; then
    echo ""
    echo "🎉 Build SUCCESS!"
    echo "📱 APK file created at:"
    echo "   $DIR/$APK_PATH"
    echo ""
    echo "To install to your C85 phone via USB:"
    echo "   adb install -r $DIR/$APK_PATH"
else
    echo "⚠️ Build finished but APK was not found at $APK_PATH"
fi
