// Stan T
import ArgumentParser
import Foundation

// MARK: - Models Command

struct ModelsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List models available to your GitHub Copilot account"
    )

    func run() async throws {
        let config = ConfigLoader.load()

        guard !config.ai.apiKey.isEmpty else {
            print("❌ Set ai.apiKey in .codecontext.json (a GitHub token with Copilot access)")
            return
        }

        let analyzer = AICodeAnalyzer(
            apiKey: config.ai.apiKey,
            model: config.ai.model,
            provider: config.ai.provider,
            baseURL: config.ai.baseURL,
            tokenURL: config.ai.tokenURL
        )

        let models = try await analyzer.listCopilotModels()
        print("🤖 \(models.count) models available:\n")
        for id in models {
            print(id == config.ai.model ? "  • \(id)  ← current" : "  • \(id)")
        }
    }
}
