#include "vc/core/types/Metadata.hpp"
#include "vc/core/types/Exceptions.hpp"

namespace fs = std::filesystem;

using namespace volcart;

// Read a json config from disk
Metadata::Metadata(fs::path fileLocation) : path_{fileLocation}
{
    // Check if file exists
    if (!fs::exists(fileLocation)) {
        auto msg = "JSON file not found: '" + fileLocation.string() + "'";
        throw IOException(msg);
    }

    // Read the entire file
    std::ifstream file(fileLocation.string(), std::ios::binary | std::ios::ate);
    if (!file) {
        auto msg = "Failed to open JSON file: '" + fileLocation.string() + "'";
        throw IOException(msg);
    }

    auto size = file.tellg();
    std::string buffer(size, '\0');
    file.seekg(0);
    file.read(buffer.data(), size);

    if (!file) {
        auto msg = "Failed to read JSON file: '" + fileLocation.string() + "'";
        throw IOException(msg);
    }
    file.close();

    // Parse JSON using glaze
    auto error = glz::read_json(json_, buffer);
    if (error) {
        auto msg = "Failed to parse JSON file '" + fileLocation.string() +
                   "': " + glz::format_error(error, buffer);
        throw IOException(msg);
    }
}

// Save the JSON file to disk
void Metadata::save(const fs::path& path)
{
    // Serialize to JSON string with pretty printing
    std::string buffer;
    auto error = glz::write<glz::opts{.prettify = true}>(json_, buffer);
    if (error) {
        auto msg = "Failed to serialize metadata: " + std::string(glz::format_error(error));
        throw IOException(msg);
    }

    // Write to file
    std::ofstream file(path.string(), std::ios::binary);
    if (!file) {
        auto msg = "Failed to open file for writing: '" + path.string() + "'";
        throw IOException(msg);
    }

    file << buffer;
    if (!file) {
        auto msg = "Failed to write JSON file: '" + path.string() + "'";
        throw IOException(msg);
    }
}

// Get the JSON as an object (throws if not an object)
glz::json_t::object_t& Metadata::asObject()
{
    if (!json_.holds<glz::json_t::object_t>()) {
        json_ = glz::json_t::object_t{};
    }
    return json_.get<glz::json_t::object_t>();
}

const glz::json_t::object_t& Metadata::asObject() const
{
    auto* obj = json_.get_if<glz::json_t::object_t>();
    if (!obj) {
        throw std::runtime_error("Metadata JSON is not an object");
    }
    return *obj;
}

// Check if a top-level key exists
bool Metadata::hasKey(const std::string& key) const
{
    auto obj = json_.get_if<glz::json_t::object_t>();
    if (!obj) {
        return false;
    }
    return obj->find(key) != obj->end();
}

// Check if a path exists (supports nested paths)
bool Metadata::hasPath(const std::string& path) const
{
    const glz::json_t* current = &json_;

    size_t start = 0;
    size_t dot = path.find('.');

    while (dot != std::string::npos) {
        std::string key = path.substr(start, dot - start);

        auto obj = current->get_if<glz::json_t::object_t>();
        if (!obj) {
            return false;
        }

        auto it = obj->find(key);
        if (it == obj->end()) {
            return false;
        }

        current = &it->second;
        start = dot + 1;
        dot = path.find('.', start);
    }

    // Check last segment
    std::string lastKey = path.substr(start);
    auto obj = current->get_if<glz::json_t::object_t>();
    if (!obj) {
        return false;
    }

    return obj->find(lastKey) != obj->end();
}

// Print compact string representation
void Metadata::printString() const
{
    std::string buffer;
    auto error = glz::write_json(json_, buffer);
    if (!error) {
        std::cout << buffer << std::endl;
    } else {
        std::cout << "Error serializing metadata: " << glz::format_error(error) << std::endl;
    }
}

// Print formatted representation
void Metadata::printObject() const
{
    std::string buffer;
    auto error = glz::write<glz::opts{.prettify = true}>(json_, buffer);
    if (!error) {
        std::cout << buffer << std::endl;
    } else {
        std::cout << "Error serializing metadata: " << glz::format_error(error) << std::endl;
    }
}