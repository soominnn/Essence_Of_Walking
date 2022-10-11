package com.walking.standardofwalking.service;

import com.walking.standardofwalking.dto.UserForm;
import com.walking.standardofwalking.entity.User;
import com.walking.standardofwalking.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    public User create(UserForm dto) {
        User user = dto.toEntity();
        if (user.getCid() != null) {
            return null;
        }
        return userRepository.save(user);
    }
}
