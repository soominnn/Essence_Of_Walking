package com.walking.standardofwalking.service;

import com.walking.standardofwalking.dto.UserForm;
import com.walking.standardofwalking.entity.User;
import com.walking.standardofwalking.repository.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Slf4j
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    public User create(UserForm dto) {
        User user = dto.toEntity();
        //비밀번호 암호화
        String encodedPassword = passwordEncoder.encode(user.getPassword());
        user.setPassword(encodedPassword);

        if (user.getCid() != null) {
            return null;
        }
        return userRepository.save(user);
    }

    //아이디로 유저 찾기
    public Optional<User> findByUserId(String userid){
        return userRepository.findAll().stream()
                .filter(m -> m.getUserid().equals(userid))
                .findFirst();
    }
    public List<User> findAll(){
        return userRepository.findAll();
    }

    //회원 삭제
    public Optional<User> delete(String userid){
        //대상 찾기
        Optional<User> target = findByUserId(userid);
        //잘못된 요청 처리
        if(target == null)
            return null;
        //대상 삭제 후 응답 반환
        userRepository.delete(target.get());
        return target;
    }

    //회원정보 수정
    public User update(String userid, UserForm dto) {
        // 1: 수정용 엔티티 생성
        User user = dto.toEntity();
        // 2: 대상 엔티티 찾기
        User target = findByUserId(userid).orElse(null);
        // 3: 잘못된 요청 처리(대상이 없는경우)
        if (target == null) {
            return null;
        }
        // 4: 업데이트 및 정상 응답(200)
        target.patch(user);
        return userRepository.save(target);
    }
}
