# Template for prat31/homebrew-tap's Formula/linpuzs.rb.
#
# The release workflow (.github/workflows/release.yml) fills in url/sha256 for
# each new PyPI release, regenerates the resource blocks with
# `brew update-python-resources`, test-installs it, and pushes it to the tap.
class Linpuzs < Formula
  include Language::Python::Virtualenv

  desc "Solves LinkedIn's daily puzzles on your Android phone over ADB"
  homepage "https://github.com/prat31/linkedin-puzzles-solver"
  url "@@SDIST_URL@@"
  sha256 "@@SDIST_SHA256@@"
  license "MIT"

  # Prebuilt by Homebrew, so they aren't compiled from source as resources.
  depends_on "numpy"
  depends_on "pillow"
  depends_on "python@3.15"

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      linpuzs talks to your phone with adb. If you don't have it yet:
        brew install --cask android-platform-tools
      Then turn on USB debugging on the phone and run: linpuzs
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/linpuzs --version")
  end
end
