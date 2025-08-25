#pragma once

/** @file */

#include <fstream>
#include <iostream>
#include <filesystem>
#include <type_traits>

#include <glaze/glaze.hpp>
#include <opencv2/core.hpp>


namespace volcart
{
class Metadata
{
public:
    Metadata() = default;
    explicit Metadata(std::filesystem::path fileLocation);
    std::filesystem::path path() const { return path_; }

    void setPath(const std::filesystem::path& path) { path_ = path; }

    void save() { save(path_); }

    void save(const std::filesystem::path& path);
    glz::json_t& json() { return json_; }
    const glz::json_t& json() const { return json_; }
    glz::json_t::object_t& asObject();
    const glz::json_t::object_t& asObject() const;
    bool hasKey(const std::string& key) const;
    bool hasPath(const std::string& path) const;
    template <typename T>
    T get(const std::string& key) const
    {
        const auto& obj = asObject();
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::runtime_error("Key '" + key + "' not found in metadata");
        }

        return extractValue<T>(it->second, key);
    }

    template <typename T>
    T getPath(const std::string& path) const
    {
        const glz::json_t* current = &json_;
        std::string fullPath;

        // Split path by dots and traverse
        size_t start = 0;
        size_t dot = path.find('.');

        while (dot != std::string::npos) {
            std::string key = path.substr(start, dot - start);
            fullPath += (fullPath.empty() ? "" : ".") + key;

            auto obj = current->get_if<glz::json_t::object_t>();
            if (!obj) {
                throw std::runtime_error("Path '" + fullPath + "' is not an object");
            }

            auto it = obj->find(key);
            if (it == obj->end()) {
                throw std::runtime_error("Path '" + fullPath + "' not found");
            }

            current = &it->second;
            start = dot + 1;
            dot = path.find('.', start);
        }

        // Handle last segment
        std::string lastKey = path.substr(start);
        fullPath += (fullPath.empty() ? "" : ".") + lastKey;

        if (start == 0) {
            // No dots, just a simple key
            return get<T>(lastKey);
        } else {
            // Get from current object
            auto obj = current->get_if<glz::json_t::object_t>();
            if (!obj) {
                throw std::runtime_error("Path '" + fullPath + "' parent is not an object");
            }

            auto it = obj->find(lastKey);
            if (it == obj->end()) {
                throw std::runtime_error("Path '" + fullPath + "' not found");
            }

            return extractValue<T>(it->second, fullPath);
        }
    }

    template <typename T>
    void set(const std::string& key, const T& value)
    {
        auto& obj = asObject();
        obj[key] = value;
    }

    template <typename T>
    void setPath(const std::string& path, const T& value)
    {
        glz::json_t* current = &json_;

        // Ensure root is an object
        if (!current->holds<glz::json_t::object_t>()) {
            *current = glz::json_t::object_t{};
        }

        // Split path and traverse/create as needed
        size_t start = 0;
        size_t dot = path.find('.');

        while (dot != std::string::npos) {
            std::string key = path.substr(start, dot - start);

            auto& obj = current->get<glz::json_t::object_t>();

            // Create intermediate object if it doesn't exist
            if (obj.find(key) == obj.end() || !obj[key].holds<glz::json_t::object_t>()) {
                obj[key] = glz::json_t::object_t{};
            }

            current = &obj[key];
            start = dot + 1;
            dot = path.find('.', start);
        }

        // Set the final value
        std::string lastKey = path.substr(start);
        auto& obj = current->get<glz::json_t::object_t>();
        obj[lastKey] = value;
    }

    template <typename T>
    void deserializeInto(T& value) const
    {
        std::string buffer;
        auto error = glz::write_json(json_, buffer);
        if (error) {
            throw std::runtime_error("Failed to serialize metadata: " + 
                                   std::string(glz::format_error(error)));
        }
        
        error = glz::read_json(value, buffer);
        if (error) {
            throw std::runtime_error("Failed to deserialize metadata: " + 
                                   std::string(glz::format_error(error, buffer)));
        }
    }

    template <typename T>
    void serializeFrom(const T& value)
    {
        std::string buffer;
        auto error = glz::write_json(value, buffer);
        if (error) {
            throw std::runtime_error("Failed to serialize value: " + 
                                   std::string(glz::format_error(error)));
        }
        
        error = glz::read_json(json_, buffer);
        if (error) {
            throw std::runtime_error("Failed to parse JSON: " + 
                                   std::string(glz::format_error(error, buffer)));
        }
    }

    void printString() const;
    void printObject() const;
private:
    template <typename T>
    static T extractValue(const glz::json_t& value, const std::string& context)
    {
        // Direct extraction for basic types
        if constexpr (std::is_same_v<T, bool>) {
            if (auto* v = value.get_if<bool>()) return *v;
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (auto* v = value.get_if<std::string>()) return *v;
        } else if constexpr (std::is_same_v<T, double>) {
            if (auto* v = value.get_if<double>()) return *v;
        } else if constexpr (std::is_floating_point_v<T>) {
            if (auto* v = value.get_if<double>()) return static_cast<T>(*v);
        } else if constexpr (std::is_integral_v<T>) {
            // JSON numbers might be stored as doubles
            if (auto* v = value.get_if<double>()) {
                return static_cast<T>(*v);
            }
        } else if constexpr (std::is_same_v<T, glz::json_t>) {
            return value;
        }
        
        // For complex types or if direct extraction failed, use serialization
        std::string buffer;
        auto error = glz::write_json(value, buffer);
        if (error) {
            throw std::runtime_error("Failed to serialize value at '" + context + "'");
        }
        
        T result{};
        error = glz::read_json(result, buffer);
        if (error) {
            throw std::runtime_error("Failed to convert value at '" + context + 
                                   "' to requested type");
        }
        return result;
    }

    glz::json_t json_ = glz::json_t::object_t{};
    std::filesystem::path path_;
};

}  // namespace volcart